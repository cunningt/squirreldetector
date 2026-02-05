package com.squirreldetector;

import ai.djl.modality.Classifications;
import ai.djl.modality.cv.output.BoundingBox;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.modality.cv.output.Rectangle;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;

/**
 * Processor that draws bounding boxes on images based on DetectedObjects results.
 */
@Component
public class BoundingBoxProcessor implements Processor {

    private static final Logger LOG = LoggerFactory.getLogger(BoundingBoxProcessor.class);

    @Override
    public void process(Exchange exchange) throws Exception {
        // Get the original image bytes (stored before DJL processing)
        byte[] imageBytes = exchange.getProperty("originalImage", byte[].class);

        // Get detection results from DJL component
        DetectedObjects detections = exchange.getIn().getBody(DetectedObjects.class);

        LOG.info("Found {} detection(s)", detections.getNumberOfObjects());
        for (Classifications.Classification item : detections.items()) {
            LOG.info("  - {}: {}%", item.getClassName(),
                String.format("%.2f", item.getProbability() * 100));
        }

        // Draw bounding boxes
        BufferedImage original = ImageIO.read(new ByteArrayInputStream(imageBytes));
        BufferedImage annotated = drawBoundingBoxes(original, detections);

        // Convert back to bytes
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(annotated, "jpg", baos);

        // Set annotated image as body
        exchange.getIn().setBody(baos.toByteArray());
        int count = detections.getNumberOfObjects();
        exchange.getIn().setHeader("DetectionCount", count);
        exchange.getIn().setHeader("SquirrelEmojis", "🐿️".repeat(count));
    }

    private BufferedImage drawBoundingBoxes(BufferedImage image, DetectedObjects detections) {
        Graphics2D g2d = image.createGraphics();
        g2d.setStroke(new BasicStroke(3));
        g2d.setFont(new Font("Arial", Font.BOLD, 18));

        int imageWidth = image.getWidth();
        int imageHeight = image.getHeight();

        for (Classifications.Classification classification : detections.items()) {
            DetectedObjects.DetectedObject item = (DetectedObjects.DetectedObject) classification;
            BoundingBox box = item.getBoundingBox();
            Rectangle rect = box.getBounds();

            int x = (int) (rect.getX() * imageWidth);
            int y = (int) (rect.getY() * imageHeight);
            int w = (int) (rect.getWidth() * imageWidth);
            int h = (int) (rect.getHeight() * imageHeight);

            // Draw box
            g2d.setColor(Color.GREEN);
            g2d.drawRect(x, y, w, h);

            // Draw label background
            String label = String.format("%s %.0f%%", item.getClassName(), item.getProbability() * 100);
            FontMetrics fm = g2d.getFontMetrics();
            int labelWidth = fm.stringWidth(label) + 10;
            int labelHeight = fm.getHeight() + 4;

            g2d.setColor(Color.GREEN);
            g2d.fillRect(x, y - labelHeight, labelWidth, labelHeight);

            // Draw label text
            g2d.setColor(Color.YELLOW);
            g2d.drawString(label, x + 5, y - 5);
        }

        g2d.dispose();
        return image;
    }
}
