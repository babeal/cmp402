import io
import json


class Parser:
    def __init__(self):
        self.buff = io.BytesIO()
        self.read_pos = 0

    def write(self, content):
        self.buff.seek(0, io.SEEK_END)
        self.buff.write(content)

    def scan_lines(self):
        self.buff.seek(self.read_pos)
        for line in self.buff.readlines():
            if line and line[-1] == ord("\n"):
                self.read_pos += len(line)
                yield line[:-1]

    def reset(self):
        self.read_pos = 0


class Llama2:
    def __init__(self, smr, endpoint_name):
        self.smr = smr
        self.endpoint_name = endpoint_name

    def run(self, prompt, temperature=0.6, top_p=0.9, max_tokens_to_sample=200):
        temperature = float(temperature)
        top_p = float(top_p)
        max_tokens_to_sample = int(max_tokens_to_sample)
        body = {
            "prompt": prompt,
            "temperature": temperature if temperature >= 0.0 and temperature <= 1.0 else 0.6,
            "top_p": top_p if top_p >= 0 and top_p <= 1.0 else 0.9,
            "max_tokens_to_sample": max_tokens_to_sample if max_tokens_to_sample < 513 else 512,
        }
        body = json.dumps(body)
        resp = self.smr.invoke_endpoint_with_response_stream(
            EndpointName=self.endpoint_name, Body=body, ContentType="application/json"
        )
        event_stream = resp["Body"]
        parser = Parser()
        output = ""
        for event in event_stream:
            parser.write(event["PayloadPart"]["Bytes"])
            for line in parser.scan_lines():
                resp = json.loads(line)
                resp_output = resp.get("outputs")[0]
                if resp_output in ["", " "]:
                    continue
                output += resp_output
                print(resp_output, end="")
