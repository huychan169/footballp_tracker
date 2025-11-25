from pipeline.config import build_default_config
from pipeline.pipeline import run_pipeline

def main():
    config = build_default_config()
    run_pipeline(config)

if __name__ == '__main__':
    main()
