curl -X POST "http://localhost:8000/generate/model1" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "你是谁？",
           "sampling_params": {
             "temperature": 0,
             "max_tokens": 512
           }
         }'


curl -X POST "http://localhost:8000/batch/model1" \
     -H "Content-Type: application/json" \
     -d '{
           "prompts": ["你好", "世界", "你是谁"],
           "sampling_params": {
             "temperature": 0.7,
             "max_tokens": 50
           }
         }'

curl -X POST "http://localhost:8000/multi-generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "人工智能的未来是什么？",
           "sampling_params": {
             "temperature": 0.9,
             "max_tokens": 200 
           },
           "models": ["model1", "model2"] 
         }'

curl -X POST "http://localhost:8000/multi-generate" \
     -H "Content-Type: application/json" \
     -d '{
           "prompt": "人工智能的未来是什么？",
           "sampling_params": {
             "temperature": 0.9,
             "max_tokens": 200 
           },
           "models": ["model1", "model1"] 
         }'

curl -X GET "http://localhost:8000/models/info"
