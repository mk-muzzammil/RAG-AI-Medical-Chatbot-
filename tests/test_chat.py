import unittest
from unittest.mock import Mock, patch

import app
from src.embeddings_api import HFInferenceEmbeddings, EmbeddingServiceError
from src.prompt import GREETING_RESPONSE, system_prompt
from langchain_core.prompts import ChatPromptTemplate


class ChatTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_greeting_does_not_require_services(self):
        with patch('app.get_chain') as chain:
            response = self.client.post('/get', data={'msg': 'hy'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, GREETING_RESPONSE)
        chain.assert_not_called()

    def test_empty_question(self):
        self.assertEqual(self.client.post('/get', data={'msg': ' '}).status_code, 400)

    def test_success(self):
        chain = Mock()
        chain.invoke.return_value = {'answer': 'A supported answer.'}
        with patch('app.get_chain', return_value=chain):
            response = self.client.post('/get', data={'msg': 'What is anemia?'})
        self.assertEqual(response.text, 'A supported answer.')
        chain.invoke.assert_called_once_with({'input': 'What is anemia?'})

    def test_billing_error_is_actionable(self):
        embeddings = HFInferenceEmbeddings(api_key='test')
        embeddings.session.post = Mock(return_value=Mock(ok=False, status_code=402))
        chain = Mock()
        chain.invoke.side_effect = lambda _: embeddings.embed_query('anemia')
        with patch('app.get_chain', return_value=chain):
            response = self.client.post('/get', data={'msg': 'What is anemia?'})
        self.assertEqual(response.status_code, 502)
        self.assertIn('credits are exhausted', response.json['error'])
        self.assertTrue(response.json['reference'])

    def test_unexpected_error_does_not_leak_details(self):
        with patch('app.get_chain', side_effect=RuntimeError('secret-token')):
            response = self.client.post('/get', data={'msg': 'anemia'})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn('secret-token', response.text)

    def test_prompt_formats_context(self):
        prompt = ChatPromptTemplate.from_messages([('system', system_prompt), ('human', '{input}')])
        result = prompt.invoke({'context': 'Book passage', 'input': 'Question'})
        self.assertIn('Book passage', result.messages[0].content)


class EmbeddingTests(unittest.TestCase):
    def embed(self, data):
        embeddings = HFInferenceEmbeddings(api_key='test')
        embeddings.session.post = Mock(return_value=Mock(ok=True, json=lambda: data))
        return embeddings.embed_query('test')

    def test_batch_and_single_vector(self):
        for data in ([[0.1] * 384], [0.1] * 384):
            self.assertEqual(len(self.embed(data)), 384)

    def test_token_pooling(self):
        self.assertEqual(self.embed([[[1.0] * 384, [3.0] * 384]]), [2.0] * 384)

    def test_invalid_responses(self):
        for data in ([], {'error': 'private'}, [[0.1] * 768], [[float('nan')] * 384]):
            with self.subTest(data_type=type(data).__name__):
                with self.assertRaises(EmbeddingServiceError):
                    self.embed(data)


if __name__ == '__main__':
    unittest.main()
