import io

import bentoml
import numpy as np
import torch
from bentoml.io import JSON, Image
from PIL import Image as PILImage

bento_model = bentoml.pytorch.get("xray_model")

runner = bento_model.to_runner()

svc = bentoml.Service(name="xray_service", runners=[runner])

@svc.api(input=Image(),output=JSON())
async def predict(img):
    b = io.BytesIO()
    
    img.save(b, "jpeg")
    
    im_bytes = b.getvalue()
        
    my_transforms = bento_model.custom_objects.get("xray_train_transforms")

    image = PILImage.open(io.BytesIO(im_bytes)).convert("RGB")

    image = torch.from_numpy(np.array(my_transforms(image).unsqueeze(0)))

    image = image.reshape(1, 3, 224, 224)
    
    batch_ret = await runner.async_run(image)
    
    return batch_ret[0].numpy()
    
    
    
    