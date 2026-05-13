.PHONY: install install-dev install-mavlink install-perception test demo mavlink-mock-demo px4-sitl-demo perception-demo tracking-demo geolocalization-demo tak-demo pipeline-demo edge-agent-demo edge-agent-px4-demo realtime-demo realtime-px4-demo realtime-webcam-demo realtime-rtsp-demo operator-demo safety-demo dashboard dashboard-local dashboard-demo demo-observation demo-abort demo-low-battery demo-link-loss demo-px4-live demo-all-safe demo-list evidence-demo quality smoke lint format typecheck all-checks pipeline-quality-demo clean-runs jetson-preflight jetson-preflight-dev jetson-run jetson-install-service jetson-status jetson-export-onnx jetson-export-engine stability-10min stability-60min

install:
	pip install -e ".[dev]"

install-dev:
	pip install -e ".[dev]"

install-mavlink:
	pip install -e ".[mavlink]"

install-perception:
	pip install -e ".[perception]"

test:
	pytest -v

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy sentinel apps

quality:
	python -m apps.tools.run_quality_gate --latest --strict

smoke:
	python -m apps.tools.run_quality_gate --no-strict

all-checks:
	$(MAKE) test
	$(MAKE) lint
	$(MAKE) typecheck

pipeline-quality-demo:
	$(MAKE) pipeline-demo
	$(MAKE) quality

demo:
	python -m apps.tools.run_mission_test --config configs/sim.yaml --mission missions/demo_search_area.json

mavlink-mock-demo:
	python -m apps.tools.run_mavlink_mock_test --config configs/sim.yaml --mission missions/demo_search_area.json

px4-sitl-demo:
	python -m apps.tools.run_px4_sitl_test --config configs/sim_px4.yaml --mission missions/demo_search_area.json

perception-demo:
	python -m apps.tools.run_perception_test --config configs/sim.yaml --backend mock --max-frames 30

perception-demo-yolo:
	python -m apps.tools.run_perception_test --config configs/sim.yaml --backend yolo --max-frames 60

tracking-demo:
	python -m apps.tools.run_tracking_test --config configs/sim.yaml --backend mock --max-frames 30

geolocalization-demo:
	python -m apps.tools.run_geolocalization_test --config configs/sim.yaml --backend mock --max-frames 30

tak-demo:
	python -m apps.tools.run_tak_bridge_test --config configs/sim.yaml --backend mock --max-frames 30 --tak-mode dry_run

pipeline-demo:
	python -m apps.tools.run_integrated_pipeline --config configs/sim.yaml --mission missions/demo_search_area.json --backend mock --max-frames 30 --tak-mode dry_run

edge-agent-demo:
	python -m apps.edge_agent.run_edge_agent --config configs/sim.yaml --mission missions/demo_search_area.json --mode mock --backend mock --tak-mode dry_run --max-frames 30

edge-agent-px4-demo:
	python -m apps.edge_agent.run_edge_agent --config configs/sim_px4.yaml --mission missions/demo_search_area.json --mode px4_sitl --backend mock --tak-mode dry_run --max-frames 30 --local-sitl-mission

realtime-demo:
	python -m apps.tools.run_realtime_loop --config configs/sim.yaml --mission-id realtime_mock --backend mock --tak-mode dry_run --max-frames 30 --target-fps 10

realtime-px4-demo:
	python -m apps.tools.run_realtime_loop --config configs/sim_px4.yaml --mission-id realtime_px4 --mode px4_sitl --backend mock --tak-mode dry_run --max-frames 30 --target-fps 5

realtime-webcam-demo:
	python3 -m apps.tools.run_realtime_loop --config configs/sim.yaml --mission-id realtime_webcam --video-source-type webcam --webcam-index 0 --backend mock --tak-mode dry_run --max-frames 100 --target-fps 10

realtime-rtsp-demo:
	python3 -m apps.tools.run_realtime_loop --config configs/sim.yaml --mission-id realtime_rtsp --video-source-type rtsp --rtsp-url "rtsp://127.0.0.1:8554/test" --backend mock --tak-mode dry_run --max-frames 100 --target-fps 10

operator-demo:
	python -m apps.tools.run_operator_gate_demo --config configs/sim.yaml --backend mock --max-frames 30 --operator-mode confirm_above_threshold

safety-demo:
	python -m apps.tools.run_safety_demo

dashboard:
	python3 -m apps.dashboard.run_dashboard --host 0.0.0.0 --port 8787 --base-dir runs

dashboard-local:
	python3 -m apps.dashboard.run_dashboard --host 127.0.0.1 --port 8787 --base-dir runs

dashboard-demo:
	@echo "Run these in separate terminals:"
	@echo "  Terminal 1: make dashboard-local"
	@echo "  Terminal 2: make realtime-demo"
	@echo "  Browser: http://127.0.0.1:8787 -> click 'Live Latest'"

demo-list:
	python3 -m apps.tools.run_demo_scenario list

demo-observation:
	python3 -m apps.tools.run_demo_scenario run observation_confirmed

demo-abort:
	python3 -m apps.tools.run_demo_scenario run operator_abort

demo-low-battery:
	python3 -m apps.tools.run_demo_scenario run low_battery_return

demo-link-loss:
	python3 -m apps.tools.run_demo_scenario run link_loss_return

demo-px4-live:
	python3 -m apps.tools.run_demo_scenario run px4_sitl_live_observation

demo-all-safe:
	python3 -m apps.tools.run_demo_scenario run-all-safe

evidence-demo:
	python3 -m apps.tools.run_demo_scenario run observation_confirmed
	@echo "Evidence package built in the run directory above under mission_package/"

clean-runs:
	rm -rf runs/*
	touch runs/.gitkeep

# --- Jetson deployment targets ---

jetson-preflight:
	python3 -m apps.edge_agent.run_jetson_agent \
		--config configs/jetson.yaml \
		--dry-run-preflight \
		--check-camera \
		--check-mavlink

jetson-preflight-dev:
	python3 -m apps.edge_agent.run_jetson_agent \
		--config configs/jetson.yaml \
		--dry-run-preflight \
		--no-strict \
		--allow-non-jetson \
		--skip-camera \
		--skip-mavlink

jetson-run:
	python3 -m apps.edge_agent.run_jetson_agent \
		--config configs/jetson.yaml \
		--mission missions/demo_search_area.json

jetson-install-service:
	@echo "Installing systemd service..."
	sudo cp deploy/systemd/ons-sentinel.service /etc/systemd/system/
	sudo systemctl daemon-reload
	sudo systemctl enable ons-sentinel
	@echo "Service installed. Start with: sudo systemctl start ons-sentinel"

jetson-status:
	sudo systemctl status ons-sentinel --no-pager || echo "Service not running or not installed"

jetson-export-onnx:
	python3 -m apps.tools.export_yolo_model --model-path yolov8n.pt --output-dir models --imgsz 640 --half

jetson-export-engine: jetson-export-onnx
	python3 -m apps.tools.export_yolo_model --model-path yolov8n.pt --output-dir models --imgsz 640 --half --build-engine

# --- Stability test targets ---

stability-10min:
	python3 -m apps.tools.run_stability_test --duration-min 10 --backend mock --target-fps 10 --no-save-frames

stability-60min:
	python3 -m apps.tools.run_stability_test --duration-min 60 --backend mock --target-fps 10 --no-save-frames
