mapping = {"0": "NORMAL", "1": "PNEUMONIA"}


# def prediction(image_tensor) -> float:
#     try:
#         best_model_path = "artifacts/01_13_2023_15_48_58/model_training/model.pt"

#         model = torch.load(best_model_path, map_location="cpu")

#         model.eval()

#         outputs = model.forward(image_tensor)

#         _, y_hat = outputs.max(1)

#         predicted_idx = str(y_hat.item())

#         label = mapping[predicted_idx]

#         return label

#     except Exception as e:
#         raise XRayException(e, sys) from e


# def predict(img,transform_path):
#     try:
#         b = io.BytesIO()

#         img.save(b, "jpeg")

#         im_bytes = b.getvalue()

#         image = image_loader(im_bytes,transform_path)

#         prediction_label = prediction(image)

#         return prediction_label

#     except Exception as e:
#         raise e
