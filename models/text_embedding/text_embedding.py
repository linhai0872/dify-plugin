
from dify_plugin import OAICompatEmbeddingModel

from models._common import openai_base_url


class ZenMuxTextEmbeddingModel(OAICompatEmbeddingModel):

    def _invoke(self, model, credentials, texts, user=None, input_type=None):
        credentials["endpoint_url"] = openai_base_url(credentials)
        return super()._invoke(model, credentials, texts, user, input_type)

    def validate_credentials(self, model, credentials):
        credentials["endpoint_url"] = openai_base_url(credentials)
        return super().validate_credentials(model, credentials)
