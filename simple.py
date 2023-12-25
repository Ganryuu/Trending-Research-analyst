import requests
import json

url = "https://run.cerebrium.ai/v3/p-ca1c1f36/falcon-7b/predict"

payload = json.dumps({"prompt": "what is the a transformer architecture"})

headers = {
  'Authorization': 'eyJhbGciOiJaWF0IjoxNzAzNTM1NDg5LCJleHAiOjIwMTkxMTE0ODl9.I1_5yZDbJB-sg-0v-Dh7rqDlXdBuUw4kSByHQlJyk77Gqx40SmmIHWnY6hNJGobtH6ONZba8VRal_Zyy8t9gV8YzB63ufpsU0EbY8fCgHchS7n0UEO7dlyAjh0oLgSOCZiER-s3S_AQlW3tCA6bNzEtbdGiRZ6Cu4IksbJu9rd--9ERoL59yb75x5706kaaviofVM7bYOgv8odtOXqHOLHXcqQkqPqgf-IA5RC4s6LjUMF3Z8fOYHylMeV6doiew6_nc8QcHpE8oGpRRi9jekgysY4_ZHiN_8LUleUJhHll9iDfi-XkksutLBr5gshirAHtiZv6EamZoRgYEJ0j3QQ',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)