
from typing import Optional
from pydantic import BaseModel, field_validator
import numpy as np
import base64


# class ImageBase(BaseModel):
#     image_array: np.ndarray
#     width: int
#     height: int

#     @field_validator("image_array", mode="before")
#     def validate_and_decode_image(cls, value):
#         """
#         Validate and decode an image array from a base64 string or pass-through if already a NumPy array.
#         """
#         if isinstance(value, str):  # Input is a base64-encoded string
#             decoded = base64.b64decode(value.encode("utf-8"))
#             return np.frombuffer(decoded, dtype=np.uint8)
#         elif isinstance(value, np.ndarray):  # Input is already a NumPy array
#             return value
#         raise ValueError("Invalid image data. Must be a NumPy array or base64 string.")

#     @field_validator("width", "height", mode="before")
#     def derive_width_height(cls, value, field, values):
#         """
#         Automatically derive width and height if not explicitly provided.
#         """
#         if field.name == "width" and "image_array" in values:
#             return values["image_array"].shape[1]  # Width is the second dimension
#         if field.name == "height" and "image_array" in values:
#             return values["image_array"].shape[0]  # Height is the first dimension
#         return value

#     def serialize_image(self) -> str:
#         """
#         Serialize the image array to a base64-encoded string.
#         """
#         if not isinstance(self.image_array, np.ndarray):
#             raise ValueError("image_array must be a NumPy array to serialize.")
#         return base64.b64encode(self.image_array.tobytes()).decode("utf-8")

#     def get_image(self, channels: int = 3) -> np.ndarray:
#         """
#         Reshape the image array using stored width, height, and optional channels.
#         """
#         return self.image_array.reshape((self.height, self.width, channels))

#     class Config:
#         json_encoders = {
#             np.ndarray: lambda array: base64.b64encode(array.tobytes()).decode("utf-8")
#         }