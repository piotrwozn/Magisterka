package com.sorai.gateway.kafka;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sorai.gateway.dto.PredictResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.test.EmbeddedKafkaBroker;
import org.springframework.kafka.test.context.EmbeddedKafka;
import org.springframework.kafka.test.utils.KafkaTestUtils;
import org.springframework.test.annotation.DirtiesContext;
import org.springframework.test.context.junit.jupiter.SpringExtension;

import java.time.Duration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.apache.kafka.clients.consumer.ConsumerConfig.AUTO_OFFSET_RESET_CONFIG;
import static org.apache.kafka.clients.consumer.ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG;
import static org.apache.kafka.clients.consumer.ConsumerConfig.GROUP_ID_CONFIG;
import static org.apache.kafka.clients.consumer.ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG;
import static org.apache.kafka.clients.consumer.ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG;
import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(SpringExtension.class)
@SpringBootTest
@EmbeddedKafka(partitions = 1, topics = {"triage-predictions"})
@DirtiesContext
class AuditProducerTest {

    @Autowired
    KafkaTemplate<String, String> kafkaTemplate;

    @Autowired
    ObjectMapper mapper;

    @Autowired
    EmbeddedKafkaBroker embeddedKafka;

    @Test
    @DisplayName("AuditProducer publishes a non-PHI event to Kafka")
    void publishesAuditEvent() throws Exception {
        AuditProducer producer = new AuditProducer(kafkaTemplate, mapper);
        // Inject @Value defaults via reflection
        var topicField = AuditProducer.class.getDeclaredField("topic");
        topicField.setAccessible(true);
        topicField.set(producer, "triage-predictions");
        var enabledField = AuditProducer.class.getDeclaredField("enabled");
        enabledField.setAccessible(true);
        enabledField.set(producer, true);

        PredictResponse response = new PredictResponse(
                0, 0.95,
                List.of(new PredictResponse.ModelPrediction("catboost", 0, List.of(0.95,0.05,0.0,0.0,0.0), 0.95)),
                new PredictResponse.MedGemmaAssessment(0, 0.9, "critical",
                        List.of("hipotensja"), List.of("SBP=95")),
                List.of(),
                new PredictResponse.ConflictInfo(false, "low", true, "ok"),
                142
        );

        producer.logPrediction(response, 142L, "192.168.1.42");
        kafkaTemplate.flush();

        // Consume back
        Map<String, Object> consumerProps = new HashMap<>();
        consumerProps.put(BOOTSTRAP_SERVERS_CONFIG, embeddedKafka.getBrokersAsString());
        consumerProps.put(GROUP_ID_CONFIG, "test-audit");
        consumerProps.put(AUTO_OFFSET_RESET_CONFIG, "earliest");
        consumerProps.put(KEY_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer");
        consumerProps.put(VALUE_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer");

        try (var consumer = new org.apache.kafka.clients.consumer.KafkaConsumer<String, String>(consumerProps)) {
            consumer.subscribe(List.of("triage-predictions"));
            var records = consumer.poll(Duration.ofSeconds(5));
            assertThat(records).isNotEmpty();
            var rec = records.iterator().next();
            String json = rec.value();
            // PHI guarantee: no clinical note in the audit event
            assertThat(json).doesNotContain("clinicalNote");
            assertThat(json).doesNotContain("reasoning");
            // Audit metadata present
            assertThat(json).contains("\"finalCategory\":0");
            assertThat(json).contains("\"alertDoctor\":true");
            assertThat(json).contains("\"processingMs\":142");
        }
    }

    @Test
    @DisplayName("AuditProducer is a no-op when audit.enabled=false")
    void disabledWhenAuditOff() throws Exception {
        AuditProducer producer = new AuditProducer(kafkaTemplate, mapper);
        var enabledField = AuditProducer.class.getDeclaredField("enabled");
        enabledField.setAccessible(true);
        enabledField.set(producer, false);

        PredictResponse response = new PredictResponse(
                0, 0.95, List.of(),
                new PredictResponse.MedGemmaAssessment(0, 0.9, "", List.of(), List.of()),
                List.of(),
                new PredictResponse.ConflictInfo(false, "low", false, ""),
                100
        );
        // Should not throw even though we don't set the topic
        producer.logPrediction(response, 100L, "0.0.0.0");
    }
}
