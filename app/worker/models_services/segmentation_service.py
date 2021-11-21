class SegmentationService:
    def __init__(self):
        pass


segmentation_service = SegmentationService(os.path.join(paths.MODELS_DIR, 'segmentation_weights.pt'))
