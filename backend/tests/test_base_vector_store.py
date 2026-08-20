import pytest

from app.vector_store.base_vector_store import BaseVectorStore


def test_base_vector_store_cannot_be_instantiated():

    with pytest.raises(TypeError):

        BaseVectorStore()