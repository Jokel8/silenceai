WORK_DIR = "SilenceAI/training"
os.chdir(WORK_DIR)

input_dir = "rawData/phoenix"
output_dir = "keypoints/phoenix"

HandKeypointsExtractor(input_dir, output_dir).extract()
print("\nVerarbeitung abgeschlossen!")
