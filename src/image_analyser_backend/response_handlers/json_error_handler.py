from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.status import HTTP_422_UNPROCESSABLE_CONTENT, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_200_OK

# exception handler: wrapper matches the general ExceptionHandler signature
async def custom_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, RequestValidationError):
        return await __validation_exception(request, exc)
    
    # fallthrough: anything else → 500 (generic)
    return await __generic_exception_handler(request, exc)

async def __validation_exception(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=HTTP_200_OK,
        content={
			"status_code": HTTP_422_UNPROCESSABLE_CONTENT,
            "message": "Validation error occurred",
            "detail": exc.errors(),
        },
    )

async def __generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status_code": HTTP_500_INTERNAL_SERVER_ERROR,
            "message": "Internal server error",
            "detail": str(exc),
        },
    )