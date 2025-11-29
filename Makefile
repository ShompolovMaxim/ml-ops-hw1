.PHONY: run-k8s run-k8s-build run-k8s-deploy run-clearml stop-all

run-k8s: run-k8s-build run-k8s-deploy run-clearml
	kubectl get pods

run-k8s-build:
	docker build -t ml-api:latest .
	docker build -t ml-dashboard:latest -f Dockerfile.dashboard .
	minikube image load ml-api:latest
	minikube image load ml-dashboard:latest

run-k8s-deploy:
	kubectl apply -f k8s/dvc-persistent-storage.yaml
	kubectl apply -f k8s/minio.yaml
	kubectl apply -f k8s/clearml-ml-api-secret.yaml
	kubectl apply -f k8s/api-deployment.yaml
	kubectl apply -f k8s/dashboard-deployment.yaml
	kubectl wait --for=condition=ready pod -l app=ml-api --timeout=120s
	kubectl wait --for=condition=ready pod -l app=ml-dashboard --timeout=120s

run-clearml:
	kubectl get namespace clearml >NUL 2>&1 || kubectl create namespace clearml
	kubectl apply -f k8s/clearml-secure-configmap.yaml -n clearml
	kubectl apply -f k8s/clearml-config.yaml -n clearml
	kubectl apply -f k8s/clearml-apiserver-alias.yaml -n clearml
	kubectl apply -f k8s/clearml-fileserver-alias.yaml -n clearml
	kubectl apply -f k8s/clearml-deploy.yaml
	kubectl wait --for=condition=ready pod -n clearml --all --timeout=300s

stop-all:
	kubectl delete -f k8s/clearml-deploy.yaml
	kubectl delete deployment ml-api ml-dashboard minio
	kubectl delete svc ml-api ml-dashboard minio
	kubectl delete pvc dvc-cache-pvc