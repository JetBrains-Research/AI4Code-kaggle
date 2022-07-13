class ModelLoader:
    @staticmethod
    def load_auto_model(model_class, checkpoint):
        model = model_class.load_from_checkpoint(checkpoint)
        return model
