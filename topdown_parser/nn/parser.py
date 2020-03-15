from typing import Dict, List, Any, Optional

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Embedding, Seq2SeqEncoder
from allennlp.nn.util import get_text_field_mask, get_final_encoder_states
from torch.nn import Dropout

from topdown_parser.dataset_readers.amconll_tools import AMSentence
from topdown_parser.nn.ContextProvider import ContextProvider
from topdown_parser.nn.DecoderCell import DecoderCell
from topdown_parser.nn.EdgeLabelModel import EdgeLabelModel
from topdown_parser.nn.EdgeModel import EdgeModel
from topdown_parser.nn.utils import get_device_id
from topdown_parser.transition_systems.transition_system import TransitionSystem


@Model.register("topdown")
class TopDownDependencyParser(Model):

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoder: DecoderCell,
                 edge_model: EdgeModel,
                 edge_label_model: EdgeLabelModel,
                 transition_system : TransitionSystem,
                 context_provider : Optional[ContextProvider] = None,
                 pos_tag_embedding: Embedding = None,
                 lemma_embedding: Embedding = None,
                 ne_embedding: Embedding = None,
                 input_dropout: float = 0.0
                 ):
        super().__init__(vocab)
        self.context_provider = context_provider
        self.transition_system = transition_system
        self.decoder = decoder
        self.edge_model = edge_model
        self.edge_label_model = edge_label_model
        self.encoder = encoder
        self.pos_tag_embedding = pos_tag_embedding
        self.lemma_embedding = lemma_embedding
        self.ne_embedding = ne_embedding
        self.text_field_embedder = text_field_embedder

        self._input_dropout = Dropout(input_dropout)

        self.encoder_output_dim = encoder.get_output_dim()
        self.decoder_output_dim = self.encoder_output_dim

        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, self.encoder_output_dim]), requires_grad=True)

        self.head_decisions_correct = 0
        self.decisions = 0

    def forward(self, words: Dict[str, torch.Tensor],
                pos_tags: torch.LongTensor,
                lemmas: torch.LongTensor,
                ner_tags: torch.LongTensor,
                metadata: List[Dict[str, Any]],
                order_metadata: List[Dict[str, Any]],
                seq: Optional[torch.Tensor] = None,
                context : Optional[Dict[str, torch.Tensor]] = None,
                active_nodes : Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                label_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        batch_size, seq_len = pos_tags.shape
        # Encode the input:
        state = self.encode(words, pos_tags, lemmas, ner_tags)  # shape (batch_size, seq_len, encoder_dim)

        # Initialize decoder
        self.decoder.reset_cell(batch_size, get_device_id(pos_tags))
        self.decoder.set_hidden_state(
            get_final_encoder_states(state["encoded_input"], state["input_mask"], self.encoder.is_bidirectional()))

        sentences = [ m["am_sentence"] for m in metadata]
        ret = {}
        if seq is not None and labels is not None and label_mask is not None and context is not None:
            ret["loss"] = self.compute_loss(state, seq, active_nodes, labels, label_mask, context)

        if not self.training:
            if "loss" in ret:
                #Reset decoder!
                self.decoder.reset_cell(batch_size, get_device_id(pos_tags))
                self.decoder.set_hidden_state(
                    get_final_encoder_states(state["encoded_input"], state["input_mask"], self.encoder.is_bidirectional()))

            sentences = [s.strip_annotation() for s in sentences]
            predictions = self.parse_sentences(state, metadata[0]["formalism"], sentences)
            ret["predictions"] = predictions

        return ret

    def encode(self, words: Dict[str, torch.Tensor],
               pos_tags: torch.LongTensor,
               lemmas: torch.LongTensor,
               ner_tags: torch.LongTensor) -> Dict[str, torch.Tensor]:

        embedded_text_input = self.text_field_embedder(words)
        concatenated_input = [embedded_text_input]
        if pos_tags is not None and self.pos_tag_embedding is not None:
            concatenated_input.append(self.pos_tag_embedding(pos_tags))
        elif self.pos_tag_embedding is not None:
            raise ConfigurationError("Model uses a POS embedding, but no POS tags were passed.")

        if self.lemma_embedding is not None:
            concatenated_input.append(self.lemma_embedding(lemmas))
        if self.ne_embedding is not None:
            concatenated_input.append(self.ne_embedding(ner_tags))

        if len(concatenated_input) > 1:
            embedded_text_input = torch.cat(concatenated_input, -1)
        mask = get_text_field_mask(words)  # shape (batch_size, input_len)
        embedded_text_input = self._input_dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)
        batch_size, seq_len, encoding_dim = encoded_text.shape
        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)

        # Concatenate the artificial root onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)

        mask = torch.cat([torch.ones((batch_size, 1), dtype=torch.long, device=get_device_id(encoded_text)), mask],
                         dim=1)
        assert mask.shape == (batch_size, seq_len + 1)

        return {"encoded_input": encoded_text, "input_mask": mask}

    def compute_loss(self, state: Dict[str, torch.Tensor], seq: torch.Tensor, active_nodes : torch.Tensor,
                     labels: torch.Tensor, label_mask: torch.Tensor, context : Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Computes the loss.
        :param context: a dictionary with key describing the context (parents, siblings) and values of shape (batch_size, decision seq len, *)
            with additional information for each decision.
        :param active_nodes: shape (batch_size, input_seq_len) with currently active node (e.g. top of stack)
        :param state: state of lstm
        :param seq: shape (batch_size, input_seq_len) with indices which elements to pick
        :param labels: (batch_size, input_seq_len) gold edge labels
        :param label_mask: (batch_size, input_seq_len) indicating where edge labels should be predicted
        :return: a tensor of shape (batch_size,) with the loss
        """

        batch_size, output_seq_len = seq.shape
        _, input_seq_len, encoder_dim = state["encoded_input"].shape

        self.edge_model.set_input(state["encoded_input"], state["input_mask"])
        self.edge_label_model.set_input(state["encoded_input"], state["input_mask"])

        loss = torch.zeros(batch_size, device=get_device_id(seq))

        assert torch.all(seq[:, 0] == 0), "The first node in the traversal must be the artificial root with index 0"

        if self.context_provider:
            self.transition_system.undo_one_batching(context)

        for step in range(output_seq_len - 1):
            # Retrieve current vector corresponding to current node and feed to decoder
            current_node = active_nodes[:, step]
            assert current_node.shape == (batch_size,)

            encoding_current_node = state["encoded_input"][range(batch_size), current_node]

            if self.context_provider:
                # Generate context snapshot of current time-step.
                current_context = { feature_name : tensor[:, step] for feature_name, tensor in context.items()}
                encoding_current_node = self.context_provider.forward(encoding_current_node, state, current_context)

            self.decoder.step(encoding_current_node)
            decoder_hidden = self.decoder.get_hidden_state()
            assert decoder_hidden.shape == (batch_size, self.decoder_output_dim)

            #####################
            # Predict edges
            edge_scores = self.edge_model.edge_scores(decoder_hidden)
            assert edge_scores.shape == (batch_size, input_seq_len)

            # Get target of gold edges
            target_gold_edges = seq[:, step + 1]
            assert target_gold_edges.shape == (batch_size,)

            # Compute edge existence loss
            current_mask = seq[:, step+1] >= 0 # are we already in the padding region?
            #target_gold_edges = F.relu(target_gold_edges)

            max_values, argmax = torch.max(edge_scores, dim=1) # shape (batch_size,)
            self.head_decisions_correct += torch.sum(current_mask * (target_gold_edges == argmax)).cpu().numpy()
            self.decisions += torch.sum(current_mask).cpu().numpy()

            #loss = loss + current_mask * (edge_scores[range(batch_size), target_gold_edges] - max_values) #TODO: check no log_softmax! TODO margin.
            loss = loss + current_mask * edge_scores[range(batch_size), target_gold_edges]

            #####################

            # Compute edge label scores
            edge_label_scores = self.edge_label_model.edge_label_scores(target_gold_edges, decoder_hidden)
            assert edge_label_scores.shape == (batch_size, self.edge_label_model.vocab_size)

            gold_labels = labels[:, step + 1]
            edge_loss = edge_label_scores[range(batch_size), gold_labels]
            assert edge_loss.shape == (batch_size,)

            # We don't have to predict an edge label everywhere, so apply the appropriate mask:
            loss = loss + label_mask[:, step + 1] * edge_loss
            #loss = loss + label_mask[:, step + 1] * F.nll_loss(edge_label_scores, gold_labels, reduction="none")

            #####################

        return -loss.sum() / batch_size

    def parse_sentences(self, state: Dict[str, torch.Tensor], formalism : str, sentences: List[AMSentence]) -> List[AMSentence]:
        """
        Parses the sentences.
        :param sentences:
        :param state:
        :return:
        """
        batch_size, input_seq_len, encoder_dim = state["encoded_input"].shape
        device = get_device_id(state["encoded_input"])

        self.edge_model.set_input(state["encoded_input"], state["input_mask"])
        self.edge_label_model.set_input(state["encoded_input"], state["input_mask"])

        INF = 100
        inverted_input_mask = INF * (1-state["input_mask"])

        self.transition_system.reset_parses(sentences, input_seq_len)

        output_seq_len = input_seq_len*2 + 1

        next_active_nodes = torch.zeros(batch_size, dtype=torch.long, device = device) #start with artificial root.

        valid_choices = torch.ones((batch_size, input_seq_len), device = device)
        all_selected_nodes = [torch.zeros(batch_size, device = device)]

        for step in range(output_seq_len):
            encoding_current_node = state["encoded_input"][range(batch_size), next_active_nodes]

            # Feed current node to decoder
            if self.context_provider:
                # Generate context snapshot
                current_context = self.transition_system.gather_context(next_active_nodes)
                encoding_current_node = self.context_provider.forward(encoding_current_node, state, current_context)

            self.decoder.step(encoding_current_node)
            decoder_hidden = self.decoder.get_hidden_state()
            assert decoder_hidden.shape == (batch_size, self.decoder_output_dim)

            #####################
            # Predict edges
            edge_scores = self.edge_model.edge_scores(decoder_hidden)
            assert edge_scores.shape == (batch_size, input_seq_len)

            # Apply filtering of valid choices:
            edge_scores = edge_scores - inverted_input_mask - INF*(1-valid_choices)

            selected_nodes = torch.argmax(edge_scores, dim=1)
            assert selected_nodes.shape == (batch_size,)
            all_selected_nodes.append(selected_nodes)

            #####################

            # Compute edge label scores
            edge_label_scores = self.edge_label_model.edge_label_scores(selected_nodes, decoder_hidden)
            assert edge_label_scores.shape == (batch_size, self.edge_label_model.vocab_size)

            selected_labels = torch.argmax(edge_label_scores, dim=1)
            assert selected_labels.shape == (batch_size,)

            selected_labels_str = [ self.vocab.get_token_from_index(label_id, namespace=formalism+"_labels") for label_id in selected_labels.cpu().numpy()]

            #####################

            ### Update current node according to transition system:
            next_active_nodes, valid_choices = self.transition_system.step(selected_nodes, selected_labels_str)
            assert next_active_nodes.shape == (batch_size,)
            assert valid_choices.shape == (batch_size, input_seq_len)
            valid_choices = valid_choices.to(get_device_id(edge_label_scores))

            if torch.all(valid_choices == 0):
                break



        return self.transition_system.retrieve_parses()

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        r = dict()
        if self.decisions > 0:
            r["unlabeled_head_decisions"] = self.head_decisions_correct / self.decisions * 100

        if reset:
            self.head_decisions_correct = 0
            self.decisions = 0
        return r


