from image_analyser_backend.request_schemas.config_request import ConfigUpdateRequestBody
from image_analyser_backend.response_handlers.json_response_handler import Response_SUCCESS
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from fastapi import HTTPException

class ConfigService:
    def config_update(self, input_data: ConfigUpdateRequestBody):
        # raise HTTPException(
        #     status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        #     detail="Internal server error occurred while updating config"
        # )
        return Response_SUCCESS(status_code=HTTP_400_BAD_REQUEST, message="Config setting has been updated successfully", data=input_data)
        