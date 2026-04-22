package com.lohika.morning.ml.spark.driver.service.lyrics.pipeline;

import com.lohika.morning.ml.spark.driver.service.BaseTest;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.junit.Test;
import org.springframework.beans.factory.annotation.Autowired;

import java.util.Map;

import static org.junit.Assert.*;

public class MendeleyPipelineTest extends BaseTest {

    @Autowired
    private MendeleyLogisticRegressionPipeline pipeline;

    @Test
    public void testClassify() {
        CrossValidatorModel model = pipeline.classify();
        assertNotNull("Model should not be null", model);

        Map<String, Object> stats = pipeline.getModelStatistics(model);
        assertNotNull("Stats should not be null", stats);
        assertTrue("Accuracy should be > 0",
                (Double) stats.get("Best model accuracy") > 0);

        System.out.println("Test passed! Model statistics: " + stats);
    }
}
