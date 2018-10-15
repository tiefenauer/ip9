from unittest import TestCase

import numpy as np
from hamcrest import assert_that, is_

from core.batch_generator import BatchGenerator


class DummyBatchGenerator(BatchGenerator):

    def __init__(self, batch_items, batch_size):
        super().__init__(batch_items, batch_size)

    def shuffle_entries(self):
        pass

    def extract_features(self, index_array):
        return [np.random.rand(i, 26) for i in index_array]

    def extract_labels(self, index_array):
        return [f'some label' for i in index_array]


class TestBatchGenerator(TestCase):

    def test_batch_generator_attributes(self):
        batch_items = list(range(33))
        batch_size = 16
        generator = DummyBatchGenerator(batch_items, batch_size)
        assert_that(generator.n, is_(33), f'n should reflect the number of batch items')
        assert_that(len(generator), is_(3), f'len() should reflect the number of batches')
        assert_that(len(generator[0][0]['the_input']), is_(batch_size), f'first batch should be full')
        assert_that(len(generator[1][0]['the_input']), is_(batch_size), f'second batch should be full')
        assert_that(len(generator[2][0]['the_input']), is_(1), f'last batch should be residual')

    def test_batch_generator_finite(self):
        batch_items = [1, 2, 3, 4, 5, 6, 7]
        batch_size = 3
        generator = DummyBatchGenerator(batch_items, batch_size)
        assert_that(len(generator), is_(3))
        for i, (batch_inputs, batch_outputs) in enumerate(generator):
            assert_that(batch_inputs['the_input'].ndim, is_(3))
            if i % len(generator) == len(generator) - 1:
                assert_that(batch_inputs['the_input'].shape[0], is_(1), f'last batch should be residual')
            else:
                assert_that(batch_inputs['the_input'].shape[0], is_(batch_size), 'batch should be full')
            assert_that(batch_inputs['the_input'].shape[2], is_(26))
            if i >= len(generator):
                break  # we need to break out because generator is infinite

        assert_that(generator.cur_index, is_(1), f'finite generator should be exhausted')

    def test_bath_generator_infinite(self):
        batch_items = [1, 2, 3, 4, 5, 6, 7]
        batch_size = 3
        generator = DummyBatchGenerator(batch_items, batch_size)
        assert_that(len(generator), is_(3), 'length should still reflect the number of batches')
        first_batch = generator[0]
        second_batch = generator[1]
        third_batch = generator[2]
        for i, (batch_inputs, batch_outputs) in enumerate(generator):
            if i % batch_size == 0:
                assert_that(batch_inputs['the_input'].shape, is_(first_batch[0]['the_input'].shape))
            elif i % batch_size == 1:
                assert_that(batch_inputs['the_input'].shape, is_(second_batch[0]['the_input'].shape))
            else:
                assert_that(batch_inputs['the_input'].shape, is_(third_batch[0]['the_input'].shape))
            if i > 10:
                break  # we need to break out because generator is infinite

        assert_that(i, is_(11))
        assert_that(generator.cur_index, is_(3), )
