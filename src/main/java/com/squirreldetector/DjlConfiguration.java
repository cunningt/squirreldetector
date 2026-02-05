package com.squirreldetector;

import ai.djl.Model;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Arrays;
import java.util.List;

@Configuration
public class DjlConfiguration {

    private static final Logger LOG = LoggerFactory.getLogger(DjlConfiguration.class);

    @Value("${detector.model.path}")
    private String modelPath;

    @Value("${detector.model.input-size}")
    private int inputSize;

    @Value("${detector.model.confidence-threshold}")
    private float confThreshold;

    @Value("${detector.model.nms-threshold}")
    private float nmsThreshold;

    @Value("${detector.model.class-names}")
    private String classNamesConfig;

    @Bean
    public YoloV8Translator yoloTranslator() {
        List<String> classNames = Arrays.asList(classNamesConfig.split(","));
        return new YoloV8Translator(classNames, inputSize, confThreshold, nmsThreshold);
    }

    @Bean
    public Model squirrelModel(YoloV8Translator translator) throws Exception {
        LOG.info("Loading model from: {}", modelPath);

        Criteria<Image, DetectedObjects> criteria = Criteria.builder()
            .setTypes(Image.class, DetectedObjects.class)
            .optModelPath(java.nio.file.Paths.get(modelPath))
            .optTranslator(translator)
            .optEngine("PyTorch")
            .build();

        ZooModel<Image, DetectedObjects> model = criteria.loadModel();
        LOG.info("Model loaded successfully!");
        return model;
    }
}
