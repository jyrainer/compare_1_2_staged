from waffle_hub.hub import Hub
from waffle_hub.dataset import Dataset

dataset = Dataset.load(name = "Ison_Det_1")
hub = Hub.new(
    name="Ison_Det_1_m",
    backend="ultralytics",
    task="OBJECT_DETECTION",
    model_type="yolov8",
    model_size="m",
    categories=dataset.get_category_names()
)

hub.train(dataset, epochs = 200, device = "7", batch_size = 128, image_size = 640, letter_box = True, advance_params = {'patience': 200, 'pretrained' : False})