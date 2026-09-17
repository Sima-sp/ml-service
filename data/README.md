# data/

Nada daqui é versionado no Git, só a estrutura (`.gitkeep`). Os dados são refeitos pelos
comandos do README.

| Pasta | Conteúdo | Gerado por |
|---|---|---|
| `raw/cge_alagamentos/html/AAAA/` | HTML de cada dia pesquisado no CGE | `src.collect.cge_alagamentos baixar` |
| `raw/geosampa/` | camadas WFS em GeoJSON | `src.collect.geosampa_wfs baixar` |
| `raw/cge_pluvio/` | boletins pluviométricos (baixados à mão) | navegador |
| `raw/open_meteo/<fonte>/` | chuva por célula × ano (Parquet) | `src.collect.open_meteo` |
| `raw/_cache_http*` / `_cache_nominatim.json` | caches de requisições | automático |
| `interim/eventos_cge.parquet` | alagamentos do CGE sem coordenadas | `cge_alagamentos extrair` |
| `interim/eventos_cge_geocodificados.parquet` | + lat/lon/precisão | `src.pontos geocodificar` |
| `interim/eventos_geosampa.parquet` | ocorrências com coordenadas | `geosampa_wfs extrair` |
| `interim/eventos.parquet` | união filtrada + zona | `src.pontos unificar` |
| `interim/chuva_horaria.parquet` | célula × hora | `src.collect.open_meteo` |
| `interim/celulas_chuva.csv` | centro e grade de cada célula | `src.collect.open_meteo` |
| `processed/pontos.csv` | pontos monitorados | `src.pontos selecionar` |
| `processed/dataset_<v>.parquet` (+ `_resumo.json`, `_freq_hist.parquet`, `_pontos.csv`) | dataset final | `src.dataset` |

Se a rede bloquear alguma fonte, coloque os arquivos baixados por outro meio nas mesmas pastas
e com os mesmos nomes.
