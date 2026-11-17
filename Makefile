PYTHON ?= python3
TEACHER ?= http://127.0.0.1:8081
SEEDS ?= 17 29 43 59 71
SECONDS_PER_RUN ?= 600
PREFIX ?= study
ARMS ?= B A C D E F G
QWEN = 91cad51170dc346986eccefdc2dd33a9da36ead9
LLAMA = 9048bad8d6e3a0f9f1726e238e451181c7e02613

.PHONY: dry-run demo smoke study repro model llama teacher
dry-run:
	$(PYTHON) test_twitch.py

demo:
	$(PYTHON) -m app.server --teacher $(TEACHER)

smoke:
	$(PYTHON) test_twitch.py --live --teacher $(TEACHER)

study:
	@set -eu; for seed in $(SEEDS); do \
		for arm in $(ARMS); do \
			base=results/$(PREFIX)-$$seed; \
			test ! -s "$$base-$$arm.json" || continue; \
			set --; \
			case $$arm in D) set -- --training "$$base-B.json";; C|G) set -- --fast;; esac; \
			$(PYTHON) -m twitch.run --arm "$$arm" --seed "$$seed" --seconds $(SECONDS_PER_RUN) \
				--teacher $(TEACHER) --out "$$base-$$arm.json" "$$@"; \
		done; \
	done

repro:
	$(PYTHON) -m scripts.plots --table-only
	$(PYTHON) test_twitch.py --all

model: .model/qwen.gguf

.model/qwen.gguf:
	mkdir -p .model
	curl -fL --retry 2 -o .model/qwen.tmp https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/$(QWEN)/qwen2.5-1.5b-instruct-q4_k_m.gguf
	echo "6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e  .model/qwen.tmp" | sha256sum -c
	mv .model/qwen.tmp $@

llama:
	git init -q .model/llama.cpp
	git -C .model/llama.cpp fetch --depth 1 https://github.com/ggml-org/llama.cpp $(LLAMA)
	git -C .model/llama.cpp checkout --detach FETCH_HEAD
	cmake -S .model/llama.cpp -B .model/llama.cpp/build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_OPENSSL=OFF
	cmake --build .model/llama.cpp/build --target llama-server -j 8

teacher:
	.model/llama.cpp/build/bin/llama-server -m .model/qwen.gguf --host 127.0.0.1 --port 8081 -c 1024 -t 3 -tb 3 -ngl 0 --parallel 1 --no-webui
