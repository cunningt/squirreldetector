package com.squirreldetector;

import org.apache.camel.builder.endpoint.EndpointRouteBuilder;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

/**
 * Camel route that watches an input folder for images,
 * runs a torchscript model to detect for the presence of squirrels using the DJL component,
 * and the saves bounding-box annotated images to output folder, or moves the image to 
 * a no detection folder.
 */
@Component
public class SquirrelDetectorRoute extends EndpointRouteBuilder {

    @Autowired
    private BoundingBoxProcessor boundingBoxProcessor;

    @Value("${detector.route.input-directory}")
    private String inputDirectory;

    @Value("${detector.route.output-directory}")
    private String outputDirectory;

    @Value("${detector.route.output-annotated-directory}")
    private String outputAnnotatedDirectory;

    @Value("${detector.route.nosquirrels-directory}")
    private String noSquirrelsDirectory;

    @Value("${detector.route.file-pattern}")
    private String filePattern;

    @Value("${detector.route.polling-delay}")
    private int pollingDelay;

    @Override
    public void configure() throws Exception {
        from(file(inputDirectory).noop(false).include(filePattern).delay(pollingDelay))
            .routeId("squirrel-detector-route")
            .log("📷 Processing: ${header.CamelFileName}")

            // Convert to byte[] and store for later annotation
            .convertBodyTo(byte[].class)
            .setProperty("originalImage", body())

            // Run detection using Camel DJL component
            .to("djl:cv/object_detection?model=squirrelModel&translator=yoloTranslator")

            // Split based on detection results
            .choice()
                .when(simple("${body.getNumberOfObjects} > 0"))
                    // Squirrels detected
                    .log("🐿️ SQUIRREL DETECTED in ${header.CamelFileName}!")

                    // Save original image to output folder
                    .setProperty("detectionResult", body())
                    .setBody(exchangeProperty("originalImage"))
                    .to(file(outputDirectory))
                    .log("   ✅ Original → ${header.CamelFileName}")

                    // Draw bounding boxes and save annotated image
                    .setBody(exchangeProperty("detectionResult"))
                    .process(boundingBoxProcessor)
                    .to(file(outputAnnotatedDirectory))
                    .log("   ${header.SquirrelEmojis} ${header.DetectionCount} squirrels detected, bounding boxes drawn → ${header.CamelFileName}")

                .otherwise()
                    // No squirrels detected - move to a directory for no detections
                    .log("🚫 No squirrels detected in ${header.CamelFileName}")
                    .setBody(exchangeProperty("originalImage"))
                    .to(file(noSquirrelsDirectory))
                    .log("   📁 Moved to no-detection folder")
            .end();
    }
}
