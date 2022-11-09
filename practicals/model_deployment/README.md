


1. Create venv

```console
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Check that model is working properly

```console
python demo.py
```

Results will be saved in `output` folder

3. Create model `.mar` file

```console
torch-model-archiver -f --model-name faceparser \
    --version 1.0 --serialized-file model_weights/bisenet.pth \
    --handler model_handler.py --export-path model_store \
    --extra-files model.py
```

4. Start TorchServe

```console
torchserve --start --model-store model_store --models memetxt.mar --ncs
```

5. Run request

```console
curl http://127.0.0.1:8080/predictions/faceparser -F "image=@data/musk.jpg"
```