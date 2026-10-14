import os
import unittest
from unittest.mock import Mock, patch

import app
from src.embeddings_pinecone import PineconeEmbeddings, PineconeEmbeddingError
from src.embeddings import query_embeddings


class PineconeEmbeddingTests(unittest.TestCase):
    def adapter(self):
        adapter = PineconeEmbeddings(api_key='test')
        def response(*args, **kwargs):
            size = len(kwargs['json']['inputs'])
            return Mock(ok=True, json=lambda: {
                'data': [{'values': [0.1] * 384} for _ in range(size)],
                'usage': {'total_tokens': size},
            })
        adapter.session.post = Mock(side_effect=response)
        return adapter

    def test_query_and_passage_modes(self):
        adapter = self.adapter()
        self.assertEqual(len(adapter.embed_query('test')), 384)
        self.assertEqual(adapter.session.post.call_args.kwargs['json']['parameters']['input_type'], 'query')
        adapter.embed_documents(['test'])
        self.assertEqual(adapter.session.post.call_args.kwargs['json']['parameters']['input_type'], 'passage')
        self.assertEqual(adapter.total_tokens, 2)

    def test_batching_and_empty_input(self):
        adapter = self.adapter()
        self.assertEqual(adapter.embed_documents([]), [])
        adapter.session.post.assert_not_called()
        self.assertEqual(len(adapter.embed_documents(['passage'] * 65)), 65)
        self.assertEqual(adapter.session.post.call_count, 2)

    def test_quota_failure_is_safe_and_not_retried(self):
        adapter = self.adapter()
        adapter.session.post = Mock(return_value=Mock(ok=False, status_code=429))
        with self.assertRaises(PineconeEmbeddingError) as exc:
            adapter.embed_query('test')
        self.assertEqual(exc.exception.status_code, 429)
        adapter.session.post.assert_called_once()

    def test_invalid_vectors(self):
        for values in ([0.1] * 1024, [float('nan')] * 384):
            adapter = self.adapter()
            adapter.session.post = Mock(return_value=Mock(ok=True, json=lambda: {'data': [{'values': values}]}))
            with self.assertRaises(PineconeEmbeddingError):
                adapter.embed_query('test')

    def test_pinecone_needs_no_huggingface_token(self):
        with patch.dict(os.environ, {'PINECONE_API_KEY': 'test', 'GROQ_API_KEY': 'test',
                                     'EMBEDDING_PROVIDER': 'pinecone', 'PINECONE_NAMESPACE': 'test'}, clear=True):
            self.assertEqual(app.missing_vars(), [])
            self.assertIsInstance(query_embeddings(), PineconeEmbeddings)

    def test_pinecone_requires_explicit_namespace(self):
        with patch.dict(os.environ, {'PINECONE_API_KEY': 'test', 'GROQ_API_KEY': 'test',
                                     'EMBEDDING_PROVIDER': 'pinecone'}, clear=True):
            self.assertEqual(app.missing_vars(), ['PINECONE_NAMESPACE'])

    def test_long_question_rejected_before_embedding(self):
        with patch('app.get_chain') as chain:
            response = app.app.test_client().post('/get', data={'msg': 'x' * 2001})
        self.assertEqual(response.status_code, 413)
        chain.assert_not_called()
