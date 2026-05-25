package com.sorai.gateway.api;

import com.sorai.gateway.TestFixtures;
import com.sorai.gateway.client.MlServiceClient;
import com.sorai.gateway.kafka.AuditProducer;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.reactive.AutoConfigureWebTestClient;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.context.DynamicPropertyRegistry;
import org.springframework.test.context.DynamicPropertySource;
import org.springframework.test.web.reactive.server.WebTestClient;

import java.io.IOException;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.verify;

/**
 * End-to-end test: full Spring context wired up, FastAPI is replaced by a
 * MockWebServer that returns canned JSON.
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@AutoConfigureWebTestClient
class PredictControllerIntegrationTest {

    private static MockWebServer mlMock;

    @BeforeEach
    void startMock() throws IOException {
        mlMock = new MockWebServer();
        mlMock.start();
    }

    @AfterEach
    void stopMock() throws IOException {
        mlMock.shutdown();
    }

    @DynamicPropertySource
    static void overrideMlUrl(DynamicPropertyRegistry registry) {
        registry.add("ml.base-url", () -> {
            try {
                if (mlMock == null) {
                    mlMock = new MockWebServer();
                    mlMock.start();
                }
                return mlMock.url("/").toString();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        // Disable Kafka audit for this test (no broker)
        registry.add("audit.enabled", () -> "false");
        registry.add("spring.kafka.bootstrap-servers", () -> "localhost:9092");
    }

    @Autowired
    WebTestClient client;

    @MockBean
    AuditProducer auditProducer;     // mock so we don't need a real Kafka broker

    @Test
    @DisplayName("POST /api/v1/predict — happy path proxies through to ML service")
    void predictHappyPath() {
        mlMock.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .setBody(TestFixtures.predictResponseJson()));

        client.post()
                .uri("/api/v1/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(TestFixtures.criticalRequest())
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.finalCategory").isEqualTo(0)
                .jsonPath("$.confidence").isEqualTo(0.95)
                .jsonPath("$.modelPredictions[0].modelName").isEqualTo("catboost")
                .jsonPath("$.conflict.alertDoctor").isEqualTo(true)
                .jsonPath("$.medgemma.riskFlags[0]").isEqualTo("hipotensja");

        // Audit log invoked once (mocked, doesn't actually send to Kafka)
        verify(auditProducer, atLeastOnce()).logPrediction(any(), anyLong(), anyString());
    }

    @Test
    @DisplayName("POST /api/v1/predict — invalid vitals returns 422")
    void predictInvalidVitals() {
        String invalid = """
                {"vitals":{"age":999,"temp":36.7,"hr":80,"sbp":120,"dbp":80,"rr":16,"o2":99},"clinicalNote":""}
                """;
        client.post()
                .uri("/api/v1/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(invalid)
                .exchange()
                .expectStatus().isEqualTo(422);
    }

    @Test
    @DisplayName("POST /api/v1/predict — ML 503 produces 503 to client (no cascade)")
    void predictMlDown() {
        // Exhaust retries
        for (int i = 0; i < 6; i++) {
            mlMock.enqueue(new MockResponse().setResponseCode(503));
        }
        client.post()
                .uri("/api/v1/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(TestFixtures.criticalRequest())
                .exchange()
                .expectStatus().is5xxServerError();
    }

    @Test
    @DisplayName("GET /api/v1/health — returns parsed health from ML")
    void healthEndpoint() {
        mlMock.enqueue(new MockResponse()
                .setResponseCode(200)
                .setHeader("Content-Type", MediaType.APPLICATION_JSON_VALUE)
                .setBody(TestFixtures.healthResponseJson()));

        client.get().uri("/api/v1/health")
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.status").isEqualTo("ok")
                .jsonPath("$.modelsLoaded[0]").isEqualTo("catboost")
                .jsonPath("$.ollamaReady").isEqualTo(true);
    }
}
