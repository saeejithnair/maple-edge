import os
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy_utils import database_exists, create_database, drop_database
from scripts.db.nasbench201_db_importer import NASBench201Model, add_model_to_db, import_models
from maple.db.database import create_session

TEST_DB_URL = 'postgresql://maple_admin:vip_maple@localhost/maple_test_db'

ROOT_DIR = '/home/smnair/work/nas/maple-edge/tests/assets/nasbench201_db_import/models'

class TestNASBench201Importer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if database_exists(TEST_DB_URL):
            drop_database(TEST_DB_URL)
        create_database(TEST_DB_URL)
        
        engine = create_engine(TEST_DB_URL)
        NASBench201Model.metadata.create_all(engine)

    def test_add_model_to_db(self):
        session = create_session(TEST_DB_URL)
        add_model_to_db(session, 'test_model', 'keras', 128, 'cells', None, 'path/to/test_model')

        model = session.query(NASBench201Model).filter(NASBench201Model.model_name == 'test_model').first()
        self.assertIsNotNone(model)
        self.assertEqual(model.framework, 'keras')
        self.assertEqual(model.dataset_version, 128)
        self.assertEqual(model.model_type, 'cells')
        self.assertIsNone(model.device)
        self.assertEqual(model.path, 'path/to/test_model')

    def test_import_models(self):
        session = create_session(TEST_DB_URL)
        import_models(ROOT_DIR, session)

        model_count = session.query(NASBench201Model).count()
        self.assertEqual(model_count, 4)  # Adjust this number based on the number of test files in the test_data directory

        example_model = session.query(NASBench201Model).filter(NASBench201Model.model_name == 'nats_cell_128_0.h5').first()
        self.assertIsNotNone(example_model)
        self.assertEqual(example_model.framework, 'keras')
        self.assertEqual(example_model.dataset_version, 128)
        self.assertEqual(example_model.model_type, 'cells')
        self.assertIsNone(example_model.device)
        self.assertEqual(example_model.path, os.path.join(ROOT_DIR, 'models_128', 'keras', 'cells', 'nats_cell_128_0.h5'))

    @classmethod
    def tearDownClass(cls):
        drop_database(TEST_DB_URL)

if __name__ == '__main__':
    unittest.main()
