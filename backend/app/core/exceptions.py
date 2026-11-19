"""Domain exceptions for FaceVision."""

from __future__ import annotations


class FaceVisionError(Exception):
    """Base exception for FaceVision."""

    status_code: int = 500
    error_code: str = "internal_error"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if error_code is not None:
            self.error_code = error_code


class ModelNotAvailableError(FaceVisionError):
    status_code = 400
    error_code = "model_not_available"


class ModelInitializationError(FaceVisionError):
    status_code = 500
    error_code = "model_initialization_failed"


class MediaValidationError(FaceVisionError):
    status_code = 400
    error_code = "invalid_media"


class MediaNotFoundError(FaceVisionError):
    status_code = 404
    error_code = "media_not_found"


class JobNotFoundError(FaceVisionError):
    status_code = 404
    error_code = "job_not_found"


class JobStateError(FaceVisionError):
    status_code = 409
    error_code = "invalid_job_state"


class JobCancelledError(FaceVisionError):
    status_code = 499
    error_code = "job_cancelled"


class StorageError(FaceVisionError):
    status_code = 500
    error_code = "storage_error"


class CloudConsentRequiredError(FaceVisionError):
    status_code = 403
    error_code = "cloud_consent_required"


class AuthenticationError(FaceVisionError):
    status_code = 401
    error_code = "unauthorized"
