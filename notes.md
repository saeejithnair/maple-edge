# Brainstorming

Need to collect some data before tomorrow. I need at least 10 architectures from 10 devices. So need to convert 100 models first.
So first work on conversion stuff. Start conversion to onnx, then conversion to tensorRT on each device, also convert to TFLite.

1. 16
2. 32
3. 64
4. 128
5. 224
6. 256
7. 300
8. 150
9. 200
10. 100

Then collect data for those 100 models on 6 devices. Then process that into a pipeline for VAE. 

No, first work on collection stuff. I need to design the collection pipeline in such a way such that I can I can get data for all the different features. And then the data loading, processing part takes all the data that gets loaded for all features and packs it into a NxD tensor with data for all the different elements. Okay, first things first, port all the data collection stuff from HW-NATS-Bench into the collection dir. This should help trigger my memory of what code I wrote, and what's left to write. And then circle back and see what else or how else I can add things.