import json
import pprint
import re
import time
from itertools import product
import logging

import requests
from bson.regex import Regex
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoReplicaSetClient, MongoClient
from pymongo.errors import AutoReconnect

from normalize_utils import FUNCTIONS, remove_accents

logger = logging.getLogger('query-analyzer')
logger.setLevel(logging.WARNING)
app = Flask(__name__)
CORS(app)
model_url = "http://0.0.0.0:5000/api/v1/real-estate-extraction"
database_url = "10.211.55.101:27017"
db = MongoClient(database_url)["real-estate"]
coll = db["post_prod_no_abbr_short"]

WEIGHTS = {
    "potential": 12,
    "surrounding": 10,
    "surrounding_characteristics": 8,
    "surrounding_name": 20
}


@app.route('/')
def index():
    return "Index API"

# HTTP Errors handlers


@app.errorhandler(404)
def url_error(e):
    return """
    Wrong URL!
    <pre>{}</pre>""".format(e), 404


@app.errorhandler(500)
def server_error(e):
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


def normalize_tags(tags):
    norm_val = {}
    for k, v in tags.items():
        if k == 'orientation' or k == 'legal' or k == 'transaction_type' or k == 'realestate_type' \
                or k == 'addr_ward' or k == 'addr_district' or k == 'addr_city' or k == 'position':

            norm_val[k] = list(set('_'.join(x.split())
                                   for l in v for x in FUNCTIONS[k](l)))
        elif k == 'interior_floor' or k == 'interior_room' or k == 'area' or k == 'price':
            for l in v:
                a = FUNCTIONS[k](l)
                if type(a) == dict:
                    if norm_val.get(k) is None:
                        norm_val[k] = {}
                    for ka, va in a.items():
                        if norm_val[k].get(ka) is None:
                            norm_val[k][ka] = set()
                        norm_val[k][ka].update(json.dumps(sva) for sva in va)
                elif type(a) == list:
                    if norm_val.get(k) is None:
                        norm_val[k] = set()
                    norm_val[k].update(json.dumps(va) for va in a)
            if type(norm_val[k]) == dict:
                for kk in norm_val[k]:
                    norm_val[k][kk] = list(json.loads(vk)
                                           for vk in norm_val[k][kk])
            else:
                norm_val[k] = list(json.loads(vk) for vk in norm_val[k])
        else:
            norm_val[k] = list(set('_'.join(x.split()) for x in v))
    return norm_val


@app.route('/api/v1/posts', methods=['POST'])
def analyze_query():
    req = request.json
    if isinstance(req, str):
        query = req
        limit = 50
        skip = 0
        res = requests.post(model_url, json=[query]).json()[0]
        raw_tags = res['tags']
    else:
        if req.get('string', True):
            query = req['query']
            res = requests.post(model_url, json=[query]).json()[0]
            raw_tags = res['tags']
        else:
            raw_tags = req['tags']
        limit = req.get('limit', 50)
        skip = req.get('skip', 0)
    tags = {}

    for chunk in raw_tags:
        t = chunk['type']
        c = chunk['content']
        if t == 'normal':
            continue
        if tags.get(t) is None:
            tags[t] = []
        c = c.lower().strip()
        tags[t].append(c)

    norm_tags = normalize_tags(tags)

    final_tags = norm_tags
    if final_tags.get('transaction_type'):
        final_tags['transaction_type'] = [
            'cho_thue' if x == 'can_thue' else
            'ban' if x == 'mua' else x for x in final_tags['transaction_type'] if x != 'can_tim'
        ]
        if len(final_tags['transaction_type']) == 0:
            del final_tags['transaction_type']
    with open('final_tag.json', 'w') as out:
        json.dump(final_tags, out, indent=1)
    secondary = []
    score = []
    match = []
    for k, v in final_tags.items():
        if k.startswith('surrounding') or k == 'potential':
            v = [remove_accents(re.sub(r"\W+|_", "", x)) for x in v]
            secondary.append({
                "norm_val.{}".format(k): {
                    "$in": v
                }
            })
            score.extend({
                "$cond": [
                    {
                        "$and": [
                            {
                                "$gt": ["$norm_val.{}".format(k), None]
                            },
                            
                            {
                                "$in": [x, "$norm_val.{}".format(k)]
                            }
                            
                        ]
                    },
                    WEIGHTS[k],
                    0
                ]
            } for x in v)

        elif k == 'price':
            match.append({
                "norm_val.{}".format(k): {
                    "$elemMatch": {
                        "$or": [
                            {
                                "low": {"$lte": x['high']},
                                "high":{"$gte": x['low']}
                            } for x in v
                        ]
                    }
                }
            })
        elif k == 'area':
            match.append({
                "norm_val.{}".format(k): {
                    "$elemMatch": {
                        "$or": [
                            {
                                "low.dien tich": {"$lte": x['high']['dien tich']},
                                "high.dien tich":{"$gte": x['low']['dien tich']}
                            } for x in v
                        ]
                    }
                }
            })
        elif k.startswith('interior'):
            for k1, v1 in v.items():
                match.append({
                    "norm_val.{}.{}".format(k, k1): {
                        "$elemMatch": {
                            "$or": [
                                {
                                    "low": {"$lte": x['high']},
                                    "high":{"$gte": x['low']}
                                } for x in v1
                            ]
                        }
                    }
                })
        else:
            match.append({
                "norm_val.{}".format(k): {
                    "$in": [remove_accents(x) for x in v]
                }
            })
    query = []

    if len(match) > 0:
        query.append({
            "$match": {
                "$and": match
            }
        })
    if len(secondary) > 0:
        query.append({
            "$match": {
                "$or": secondary
            }
        })
    rt = final_tags.get('realestate_type')
    project = {
        "_id": 1,
        "content": 1,
        "title": 1,
        "norm_val": 1,
        "date": 1
    }
    if rt is not None and "nha" in rt and len(rt) > 1:
        score.append({
            "$cond": [
                {
                    "$or": [
                        {"$in": [x, "$norm_val.realestate_type"]} for x in rt if x != "nha"
                    ]
                },
                25,
                0
            ]
        })
    if len(score) > 0:
        project["score"] = {
            "$add": score
        }
    else:
        project["score"] = {"$add": [0]}

    query.append({"$project": project})

    query.append({"$sort": {"score": -1, "date": -1}})

    query.append({'$skip': skip})
    query.append({"$limit": limit})
    with open('query.json', 'w') as out:
        json.dump(query, out, indent=2)

    for _ in range(60):
        try:
            res = [x for x in coll.aggregate(query)]
            break
        except AutoReconnect as e:
            logger.warning(str(e))
            time.sleep(5)
    return jsonify({
        'data': res,
        'tags': raw_tags,
        'query': query
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=4774)

