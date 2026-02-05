# Squirrel Detector

A Spring Boot + Apache Camel application that uses a YOLOv8 model to detect squirrels in images.

## Tech Stack

- **Java 17+**
- **Spring Boot 3.5.9**
- **Apache Camel 4.17.0**
- **DJL (Deep Java Library) 0.36.0** with PyTorch engine
- **YOLOv8** model in TorchScript format

## Project Structure

```
src/main/java/com/squirreldetector/
├── SquirrelDetectorApplication.java  # Spring Boot entry point
├── SquirrelDetectorRoute.java        # Camel route - orchestrates the pipeline
├── BoundingBoxProcessor.java         # Draws detection boxes on images
├── YoloV8Translator.java             # DJL translator for YOLOv8 model I/O
└── DjlConfiguration.java             # Spring config for model loading

model/best.torchscript                # Pre-trained YOLOv8 model
```

## How It Works

1. Images are dropped into `input/` folder
2. Camel polls for new images (jpg, jpeg, png)
3. Images are passed through YOLOv8 model via DJL
4. Based on detection results:
   - **Squirrel detected**: Original saved to `output-detected/`, annotated version with bounding boxes saved to `output-annotated/`
   - **No squirrel**: Image moved to `output-nodetection/`

## Configuration

Key settings in `application.properties`:

| Property | Description | Default |
|----------|-------------|---------|
| `detector.model.confidence-threshold` | Minimum confidence for detection | 0.40 |
| `detector.model.nms-threshold` | Non-max suppression threshold | 0.45 |
| `detector.model.input-size` | Model input dimensions | 640 |
| `detector.route.polling-delay` | Folder polling interval (ms) | 1000 |

## Build & Run

```bash
# Build
mvn package -DskipTests

# Run
java -jar target/squirrel-detector-camel-1.0-SNAPSHOT.jar
```

## Output Folders

- `output-detected/` - Original images where squirrels were found
- `output-annotated/` - Images with green bounding boxes drawn
- `output-nodetection/` - Images with no squirrels detected
