build:
	docker-compose build

build-yarn:
	docker-compose -f docker-compose.yarn.yml build

build-yarn-nc:
	docker-compose -f docker-compose.yarn.yml build --no-cache

build-nc:
	docker-compose build --no-cache

build-progress:
	docker-compose build --no-cache --progress=plain

down:
	docker-compose down --volumes --remove-orphans

down-yarn:
	docker-compose -f docker-compose.yarn.yml down --volumes --remove-orphans

run:
	make down && docker-compose up

run-scaled:
	make down && docker-compose up --scale spark-worker=3

run-d:
	make down && docker-compose up -d

run-yarn:
	make down-yarn && docker-compose -f docker-compose.yarn.yml up

run-yarn-scaled:
	make down-yarn && docker-compose -f docker-compose.yarn.yml up --scale spark-yarn-worker=3

stop:
	docker-compose stop

stop-yarn:
	docker-compose -f docker-compose.yarn.yml stop


submit:
	docker exec -e PYSPARK_DRIVER_PYTHON=python -e PYSPARK_PYTHON=./environment/bin/python da-spark-master \
	bash -c \
	"source /opt/spark/data/brtdl_env/bin/activate && \
	spark-submit --archives /opt/spark/data/brtdl.tar.gz#environment --master spark://spark-master:7077 --deploy-mode client \
	./apps/$(app)"

submit-da-book:
	make submit app=data_analysis_book/$(app)

submit-yarn-test:
	docker exec da-spark-yarn-master spark-submit --master yarn --deploy-mode cluster ./examples/src/main/python/pi.py

submit-yarn-cluster:
	docker exec da-spark-yarn-master spark-submit --master yarn --deploy-mode cluster ./apps/$(app)

rm-results:
	rm -r book_data/results/*
