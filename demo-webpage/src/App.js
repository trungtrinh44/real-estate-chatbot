import React, { Component } from 'react';
import './App.css';
import TextArea from './TextArea';
import { download } from './utils';
import categories from './categories.json';
import { PREDICT_API, QUERY_API } from './constants';
import axios from 'axios';
import Tabs from 'react-bootstrap/lib/Tabs';
import Tab from 'react-bootstrap/lib/Tab';
import Panel from 'react-bootstrap/lib/Panel';
class App extends Component {
  constructor() {
    super();
    categories.normal = {
      color: '#FFFFFF',
      shortcut: ' ',
    };
    this.state = {
      idx: -1,
      value: "",
      query: "",
      posts: null
    };
  }

  predict = () => {
    let {
      value,
    } = this.state;
    const newLocal = this;
    axios.post(PREDICT_API, [value]).then((res) => {
      const newData = this.combineChunk(res.data[0]['tags']);
      // console.log(data[idx].content)
      value = newData.content;
      let runs = newData.runs;
      newLocal.setState({ value, runs });
      // console.log(data[idx].content);
      // console.log(runs[idx]);
    });
  }

  combineChunk = (chunks) => {
    const content = chunks.map(i => i['content']).join(' ');
    // console.log(chunks)
    const runs = {};
    let s = 0;
    let p = null;
    chunks.forEach((chunk, idx) => {
      runs[s] = {
        type: chunk['type'],
        end: s + chunk['content'].length,
        prev: p,
      };
      p = s;
      s += chunk['content'].length;
      if (idx < chunks.length - 1) {
        runs[s] = {
          type: 'normal',
          end: s + 1,
          prev: p,
        };
        p = s;
        s += 1;
      }
    });
    s = 0;
    while (runs[s]) {
      const next = runs[runs[s].end];
      if (next && next.type === runs[s].type) {
        const nextNext = runs[next.end];
        const temp = runs[s].end;
        runs[s].end = next.end;
        if (nextNext) {
          nextNext.prev = s;
        }
        delete runs[temp];
      } else {
        s = runs[s].end;
      }
    }
    return { content, runs };
  }
  search = () => {
    const newLocal = this;
    console.log(this.state.query)
    axios.post(QUERY_API, JSON.stringify(this.state.query), {
      headers: {
        'Content-Type': 'application/json'
      }
    }).then((res) => {
      this.setState({ posts: res.data['data'] })
    });
  }
  render() {
    const {
      idx, value, runs, query, posts
    } = this.state;
    return (
      <div className="App container">
        <Tabs id="uncontrolled-tab-example">
          <Tab eventKey={1} title="Model Demo">
            <textarea
              className="input-text col-xs-12 col-sm-12 col-md-12 col-lg-12"
              value={value}
              onChange={e => this.setState(
                { value: e.target.value, runs: null }
              )}
              rows={7} />
            <button
              type="button"
              className="btn btn-default"
              key="predict"
              onClick={this.predict}
              disabled={value === ""}
            >
              Predict
          </button>
            {value ? (
              <TextArea
                key="text-area"
                id={`article-${idx}`}
                text={value}
                categories={categories}
                runs={runs}
                onSaved={() => { }}
              />
            ) : null}
          </Tab>
          <Tab eventKey={2} title="Search Demo">
            <textarea
              className="input-text col-xs-12 col-sm-12 col-md-12 col-lg-12"
              value={query}
              onChange={e => this.setState(
                { query: e.target.value }
              )}
              rows={3} />
            <button
              disabled={query === ""}
              type="button"
              className="btn btn-default"
              key="search"
              onClick={this.search}
            >
              Search
          </button>
            {posts && posts.map(({ _id, title, content }) => <Panel key={_id}>
              <Panel.Heading>{title}</Panel.Heading>
              <Panel.Body>{content}</Panel.Body>
            </Panel>)}
          </Tab>
        </Tabs>
      </div>
    );
  }
}

export default App;
