import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from store_index import prepare_book
from src.embeddings import query_embeddings


class IndexingTests(unittest.TestCase):
    def test_repeatable_ids_and_page_metadata(self):
        reader = Mock(pages=[Mock(extract_text=lambda: ""),
                             Mock(extract_text=lambda: "A medical reference passage. " * 100)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "book.pdf"
            path.write_bytes(b"test PDF identity")
            with patch('store_index.PdfReader', return_value=reader):
                docs, ids, report = prepare_book(path)
                _, repeated_ids, repeated_report = prepare_book(path)
                _, changed_ids, _ = prepare_book(path, chunk_size=800)
                _, other_model_ids, _ = prepare_book(path, model='llama-text-embed-v2')
        self.assertEqual(ids, repeated_ids)
        self.assertEqual(report, repeated_report)
        self.assertNotEqual(ids[0], changed_ids[0])
        self.assertNotEqual(ids[0], other_model_ids[0])
        self.assertEqual(len(set(ids)), len(docs))
        self.assertEqual(report['pages'], 2)
        self.assertEqual(report['text_pages'], 1)
        self.assertTrue(all(doc.metadata['page'] == 2 for doc in docs))
        self.assertTrue(all(doc.metadata['source'] == 'book.pdf' for doc in docs))

    def test_empty_pdf_requires_ocr(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.pdf"
            path.write_bytes(b"blank PDF identity")
            with patch('store_index.PdfReader', return_value=Mock(pages=[Mock(extract_text=lambda: '')])):
                with self.assertRaisesRegex(ValueError, 'OCR'):
                    prepare_book(path)

    def test_local_provider_does_not_use_hosted_embeddings(self):
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'local', 'VERCEL': ''}):
            with patch('src.embeddings.local_embeddings', return_value='local'):
                self.assertEqual(query_embeddings(), 'local')

    def test_vercel_rejects_local_provider(self):
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'local', 'VERCEL': '1'}):
            with self.assertRaisesRegex(ValueError, 'Vercel'):
                query_embeddings()

    def test_hosted_model_must_match_index(self):
        with patch.dict(os.environ, {'EMBEDDING_PROVIDER': 'huggingface', 'EMBEDDING_MODEL': 'different-model'}):
            with self.assertRaisesRegex(ValueError, 'match'):
                query_embeddings()
