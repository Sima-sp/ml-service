# Integração com o backend (Java/Spring) — dias 14–15

> **Sugestões** de implementação para o monólito modular. Os nomes seguem os domínios
> existentes (`prediction`, `reading`, `alert`, `sensor`). Confira no código antes de aplicar:
> este repositório não tem acesso ao backend. Mudança de contrato = decisão de arquitetura
> (registrar também no README/ADR do backend).

## 1. docker-compose do projeto

```yaml
  ml-service:
    build: ./ml-service
    container_name: sima-ml-service
    ports: ["5000:5000"]
    environment:
      VERSAO_MODELO: v1
      NIVEL_RISCO_COM_ACENTO: "false"
    volumes:
      - ./ml-service/models:/app/models:ro
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"]
      interval: 30s
      retries: 3

  backend:
    environment:
      AI_ENABLED: "true"
      AI_SERVICE_URL: http://ml-service:5000/predict
      AI_SERVICE_BATCH_URL: http://ml-service:5000/predict/batch
    depends_on:
      ml-service:
        condition: service_healthy
```

## 2. DTOs do gateway (`prediction/infrastructure`)

```java
public record FloodPredictionRequest(
        Long sensorId,
        double latitude,
        double longitude,
        OffsetDateTime timestamp,     // opcional
        Double nivelAgua,             // null quando o sensor está sem leitura
        Double porcentagemLixo,
        Double chuvaMm                // legado; ver docs/recomendacoes.md (R8)
) {}

public record FloodPredictionResponse(
        Long sensorId,
        Double probabilidadeAlagamento,   // Double: vem null no modo REGRAS (R7)
        RiskLevel nivelRisco,
        RiskLevel nivelRiscoModelo,
        int janelaHoras,
        OffsetDateTime geradaEm,
        OffsetDateTime validaAte,
        OffsetDateTime horaReferencia,
        boolean medicaoTransbordando,
        boolean ajusteSensorAplicado,
        List<String> motivosAjuste,
        boolean semLeituraSensor,
        String fonteChuva,
        String origem,                     // MODELO | REGRAS
        String modeloVersao,
        List<String> avisos
) {}

public record BatchItem(Long sensorId, boolean sucesso, FloodPredictionResponse previsao,
                        String erro, String codigoErro) {}
```

- `RiskLevel` precisa ter `BAIXO, MEDIO, ALTO, CRITICO` (sem acento, R6).
- Configure o Jackson para **ignorar campos desconhecidos**
  (`spring.jackson.deserialization.fail-on-unknown-properties=false`), assim campos novos do
  serviço não quebram o backend.

## 3. Gateway com timeout e fallback

```java
@Component
@ConditionalOnProperty(name = "ai.enabled", havingValue = "true")
class HttpFloodPredictionGateway implements FloodPredictionGateway {

    private final RestClient client;   // timeout de leitura 2 s (por leitura), 10 s (lote)
    private final RiskClassificationService regras;

    public PredictionResult predict(FloodPredictionRequest req) {
        try {
            var resp = client.post().uri(props.url()).body(req).retrieve()
                             .body(FloodPredictionResponse.class);
            return PredictionResult.fromIa(resp);
        } catch (RestClientException e) {          // timeout, 5xx, 503 CHUVA_INDISPONIVEL
            log.warn("IA indisponível, usando regras: {}", e.getMessage());
            return PredictionResult.fromRegras(regras.classify(req));   // fonte = REGRAS
        }
    }
}
```

Sugestão: um *circuit breaker* simples (contador em memória). Depois de 5 falhas seguidas, vai
direto para as regras por 60 s, sem esperar o timeout a cada leitura.

## 4. Tabela `prediction` (Flyway, MySQL)

`src/main/resources/db/migration/V{n}__create_prediction.sql`:

```sql
CREATE TABLE prediction (
    id                      BIGINT AUTO_INCREMENT PRIMARY KEY,
    sensor_id               BIGINT       NOT NULL,
    reading_id              BIGINT       NULL,               -- leitura que disparou (se houver)
    gerada_em               DATETIME(3)  NOT NULL,
    valida_ate              DATETIME(3)  NOT NULL,
    hora_referencia         DATETIME(3)  NULL,
    probabilidade           DECIMAL(6,5) NULL,               -- NULL quando fonte = REGRAS
    nivel_risco             VARCHAR(10)  NOT NULL,
    nivel_risco_modelo      VARCHAR(10)  NULL,
    medicao_transbordando   BOOLEAN      NOT NULL DEFAULT FALSE,
    ajuste_sensor_aplicado  BOOLEAN      NOT NULL DEFAULT FALSE,
    sem_leitura_sensor      BOOLEAN      NOT NULL DEFAULT FALSE,
    fonte_chuva             VARCHAR(20)  NULL,
    fonte                   VARCHAR(10)  NOT NULL,           -- IA | REGRAS
    modelo_versao           VARCHAR(20)  NULL,
    CONSTRAINT fk_prediction_sensor  FOREIGN KEY (sensor_id)  REFERENCES sensor(id),
    CONSTRAINT fk_prediction_reading FOREIGN KEY (reading_id) REFERENCES reading(id)
);
CREATE INDEX ix_prediction_sensor_gerada ON prediction (sensor_id, gerada_em DESC);
```

(Os nomes das tabelas `sensor`/`reading` precisam ser conferidos no mapeamento JPA atual.)

## 5. Agendador

```java
@Scheduled(fixedDelayString = "${ai.scheduler.delay-ms:600000}")   // 10 min
void atualizarPrevisoes() {
    var sensores = sensorRepository.findByStatusIn(List.of(ACTIVE, MAINTENANCE));
    var itens = sensores.stream().map(s -> {
        var ultima = s.getStatus() == ACTIVE ? readingRepository.findTopBySensorOrderByTimestampDesc(s) : null;
        return FloodPredictionRequest.of(s, ultima);   // MAINTENANCE -> sem nivelAgua/porcentagemLixo
    }).toList();
    // POST /predict/batch (1 chamada); para itens com sucesso=false -> regras
    // salvar Prediction e chamar AlertService
}
```

Recálculo por leitura nova: no `ReadingService`, depois de salvar a leitura, chamar
`/predict` (síncrono agora; *consumer* RabbitMQ depois).

## 6. Alertas

- Criar alerta quando `nivelRisco` for ALTO ou CRITICO **e** não existir alerta ACTIVE para o sensor.
- Sugestão: adicionar `origem` ao `Alert` (**PREVISTO** × **MEDIDO**). `medicaoTransbordando = true`
  gera um alerta MEDIDO. Se já existir um alerta PREVISTO ativo, ele passa a MEDIDO em vez de
  duplicar.
- Resolver automaticamente um alerta PREVISTO quando a previsão válida mais recente voltar a
  BAIXO/MEDIO (opcional; manter a resolução manual para a Defesa Civil).

## 7. Endpoint público do mapa

`GET /api/public/mapa-risco` (sem autenticação; só dados agregados):

```json
[
  {
    "sensorId": 12, "latitude": -23.5152, "longitude": -46.5841, "bairro": "Vila Maria",
    "statusPrevisao": "VALIDA",
    "previsao": { "probabilidade": 0.72, "nivelRisco": "ALTO", "geradaEm": "...", "validaAte": "...", "fonte": "IA" },
    "medido": { "nivelAgua": 82.0, "porcentagemLixo": 64.0, "lidoEm": "...", "semLeitura": false }
  }
]
```

`statusPrevisao`: `VALIDA` (agora ≤ validaAte), `DESATUALIZADA` (passou do prazo) ou
`SEM_PREVISAO`. O app mostra o número só em `VALIDA`. A consulta pega a última `prediction`
por sensor (índice `(sensor_id, gerada_em)`).

## 8. Checklist de "pronto" (briefing, item 5)

- [ ] `AI_ENABLED=true` e o serviço sobe no compose
- [ ] DTOs aceitam `probabilidadeAlagamento = null` e campos desconhecidos
- [ ] timeout de 2 s + fallback para regras com `fonte = REGRAS`
- [ ] tabela `prediction` criada via Flyway
- [ ] agendador de 10 min usando `/predict/batch`
- [ ] recálculo ao receber leitura
- [ ] alerta ALTO/CRITICO sem duplicar
- [ ] `GET /api/public/mapa-risco` com `statusPrevisao`
- [ ] limiares do `RiskClassificationService` copiados para `src/baseline.py`
