
from typing import Optional
from pydantic import BaseModel, ConfigDict, field_validator
import numpy as np
import base64

class Image(BaseModel):
    image_array: np.ndarray
    width: int
    height: int

    model_config = ConfigDict(arbitrary_types_allowed=True)


    def get_image(self, channels: int = 1) -> np.ndarray:
        """
        Reshape the image array using stored width, height, and optional channels.
        """
        return self.image_array.reshape((self.height, self.width, channels))
