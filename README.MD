### Запуск
* запустить `minikube`
* `make run-k8s`

### Остановка
* `make stop-all`

### Доступ к дашборду
* `minikube service ml-dashboard --url`
* перейти по url

### Что сделано
* HTTP эндпоинты:
    * `/status` - состояние сервиса
    * `/models/available` - типы моделей, которые можно обучить
    * `/datasets/upload` - загрузить датасет (`curl.exe -X POST ... -F "file=@4.csv"`)
    * `/datasets/download/{filename}` - выгрузить датасет
    * `/models/train` - обучить модель
    * `/models/{model_id}/retrain` - переобучить модель
    * `/models/{model_id}/predict` - инференс
    * `/models` - обученные модели
    * `/models/{model_id}/info` - информация о модели
    * `/models/{model_id}` - удалить модель
    * `/datasets` - список датасетов
    * `/datasets/{filename}` - удалить датасет
    * `/docs` - fast api документация
* clear ml в minikube
* датасеты хранятся в s3 minio и загружаются туда через dvc
* дашборд на streamlit
* grpc не реализовано
* при обучении модель сохраняется в clear ml
* при инференсе - ошибка, не получается загрузить из clear ml
