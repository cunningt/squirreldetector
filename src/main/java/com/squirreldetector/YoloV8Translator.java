package com.squirreldetector;

import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.List;

/**
 * Translator for YOLOv8 TorchScript models.
 * Handles preprocessing images and postprocessing detections.
 */
public class YoloV8Translator implements Translator<Image, DetectedObjects> {

    private final int inputSize;
    private final float confThreshold;
    private final float nmsThreshold;
    private final List<String> classNames;

    public YoloV8Translator(List<String> classNames, int inputSize, float confThreshold, float nmsThreshold) {
        this.classNames = classNames;
        this.inputSize = inputSize;
        this.confThreshold = confThreshold;
        this.nmsThreshold = nmsThreshold;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Image input) {
        NDManager manager = ctx.getNDManager();

        // Convert to NDArray and resize
        NDArray array = input.toNDArray(manager);

        // Squeeze any extra dimensions to ensure we have HWC format
        while (array.getShape().dimension() > 3) {
            array = array.squeeze(0);
        }

        array = NDImageUtils.resize(array, inputSize, inputSize);

        // Normalize to 0-1 and transpose to CHW format
        array = array.transpose(2, 0, 1);  // HWC -> CHW
        array = array.div(255.0f);
        array = array.toType(DataType.FLOAT32, false);
        // Note: DJL's batchPredict adds the batch dimension automatically

        return new NDList(array);
    }

    @Override
    public DetectedObjects processOutput(TranslatorContext ctx, NDList list) {
        // YOLOv8 output shape: [1, 4 + num_classes, 8400]
        NDArray output = list.get(0);
        
        // Transpose to [8400, 4 + num_classes]
        output = output.squeeze(0).transpose();
        
        long numPredictions = output.getShape().get(0);
        int numClasses = classNames.size();
        
        List<String> retClasses = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();

        // Process each prediction
        for (int i = 0; i < numPredictions; i++) {
            NDArray prediction = output.get(i);
            float[] predArray = prediction.toFloatArray();
            
            // First 4 values are box coordinates (x_center, y_center, width, height)
            float xCenter = predArray[0];
            float yCenter = predArray[1];
            float width = predArray[2];
            float height = predArray[3];
            
            // Find best class score
            float maxScore = 0;
            int maxIndex = 0;
            for (int j = 0; j < numClasses; j++) {
                float score = predArray[4 + j];
                if (score > maxScore) {
                    maxScore = score;
                    maxIndex = j;
                }
            }
            
            // Filter by confidence threshold
            if (maxScore < confThreshold) {
                continue;
            }
            
            // Convert from center format to corner format and normalize to 0-1
            float x1 = (xCenter - width / 2) / inputSize;
            float y1 = (yCenter - height / 2) / inputSize;
            float w = width / inputSize;
            float h = height / inputSize;
            
            // Clamp to valid range
            x1 = Math.max(0, Math.min(1, x1));
            y1 = Math.max(0, Math.min(1, y1));
            w = Math.max(0, Math.min(1 - x1, w));
            h = Math.max(0, Math.min(1 - y1, h));
            
            retClasses.add(classNames.get(maxIndex));
            retProbs.add((double) maxScore);
            retBB.add(new Rectangle(x1, y1, w, h));
        }
        
        // Apply NMS
        return nms(retClasses, retProbs, retBB);
    }

    private DetectedObjects nms(List<String> classes, List<Double> probs, List<BoundingBox> boxes) {
        List<String> retClasses = new ArrayList<>();
        List<Double> retProbs = new ArrayList<>();
        List<BoundingBox> retBB = new ArrayList<>();
        
        // Sort by probability (descending)
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < probs.size(); i++) {
            indices.add(i);
        }
        indices.sort((a, b) -> Double.compare(probs.get(b), probs.get(a)));
        
        boolean[] suppressed = new boolean[indices.size()];
        
        for (int i = 0; i < indices.size(); i++) {
            int idx = indices.get(i);
            if (suppressed[i]) continue;
            
            retClasses.add(classes.get(idx));
            retProbs.add(probs.get(idx));
            retBB.add(boxes.get(idx));
            
            Rectangle box1 = (Rectangle) boxes.get(idx);
            
            // Suppress overlapping boxes
            for (int j = i + 1; j < indices.size(); j++) {
                int jdx = indices.get(j);
                if (suppressed[j]) continue;
                
                // Only suppress same class
                if (!classes.get(idx).equals(classes.get(jdx))) continue;
                
                Rectangle box2 = (Rectangle) boxes.get(jdx);
                float iou = computeIoU(box1, box2);
                if (iou > nmsThreshold) {
                    suppressed[j] = true;
                }
            }
        }
        
        return new DetectedObjects(retClasses, retProbs, retBB);
    }

    private float computeIoU(Rectangle box1, Rectangle box2) {
        double x1 = Math.max(box1.getX(), box2.getX());
        double y1 = Math.max(box1.getY(), box2.getY());
        double x2 = Math.min(box1.getX() + box1.getWidth(), box2.getX() + box2.getWidth());
        double y2 = Math.min(box1.getY() + box1.getHeight(), box2.getY() + box2.getHeight());
        
        double intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
        double area1 = box1.getWidth() * box1.getHeight();
        double area2 = box2.getWidth() * box2.getHeight();
        double union = area1 + area2 - intersection;
        
        return (float) (intersection / union);
    }
}
