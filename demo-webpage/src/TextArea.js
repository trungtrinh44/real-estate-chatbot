import React, { Component } from 'react';
import PropTypes from 'prop-types';

import TinyColor from './tinycolor';
import { trimNewLine, getSelectionText, checkParentRelation } from './utils';
import './TextArea.css';

export default class TextArea extends Component {
  static propTypes = {
    categories: PropTypes.objectOf(PropTypes.shape({
      color: PropTypes.string.isRequired,
      shortcut: PropTypes.string.isRequired,
    })).isRequired,
    runs: PropTypes.objectOf(PropTypes.shape({
      end: PropTypes.number.isRequired,
      type: PropTypes.string.isRequired,
      prev: PropTypes.number,
    })),
    text: PropTypes.string.isRequired,
    id: PropTypes.string.isRequired,
    onSaved: PropTypes.func.isRequired,
  };
  static defaultProps = {
    runs: null,
  };
  constructor(props) {
    super(props);
    const text = props.text
      .split('\n')
      .map(trimNewLine)
      .join('\n');
    this.state = {
      text,
      runs: props.runs
        ? props.runs
        : {
          0: {
            end: text.length,
            type: 'normal',
            prev: null,
          },
        },
    };
  }

  componentWillMount() {
    Object.keys(this.props.categories).forEach((x) => {
      const listener = this.handleKeyDown(x);
      document.addEventListener('keydown', listener);
      this.shortcutListener.push(listener);
    });
  }

  componentWillReceiveProps(nextProps) {
    const text = nextProps.text
      .split('\n')
      .map(trimNewLine)
      .join('\n');
    this.setState({
      text,
      runs: nextProps.runs
        ? nextProps.runs
        : {
          0: {
            end: text.length,
            type: 'normal',
            prev: null,
          },
        },
    });
  }

  componentWillUnmount() {
    this.shortcutListener.forEach(listener =>
      document.removeEventListener('keydown', listener));
    this.shortcutListener = [];
  }

  container = null;
  shortcutListener = [];

  handleKeyDown = name => (e) => {
    if (e.key.toLowerCase() === this.props.categories[name].shortcut.toLowerCase()) {
      this.handleTextSelected(name);
    }
  };

  createButton = (name, idx) => (
    <button
      type="button"
      className="btn btn-default"
      onClick={() => this.handleTextSelected(name)}
      key={`${name}-${idx}`}
      style={{
        backgroundColor: this.props.categories[name].color,
        color: TinyColor(this.props.categories[name].color).getBrightness() < 196 ? 'white' : 'black',
      }}
    >
      {`${name} (${this.props.categories[name].shortcut})`}
    </button>
  );
  handleTextSelected = (name) => {
    // console.log(this.container.contains)
    const range = getSelectionText();
    if (!range) return;
    if (!checkParentRelation(this.container, range.commonAncestorContainer)) {
      return;
    }
    const {
      startContainer, endContainer, startOffset, endOffset,
    } = range;
    if (startOffset === endOffset) return;
    const startContainerId = startContainer.parentNode.id.split('-');
    const endContainerId = endContainer.parentNode.id.split('-');
    const startRunIdx = parseInt(startContainerId[0], 10);
    const startRunLineOffset = parseInt(startContainerId[1], 10);
    const endRunIdx = parseInt(endContainerId[0], 10);
    const endRunLineOffset = parseInt(endContainerId[1], 10);
    const startIdx = startRunIdx + startRunLineOffset + startOffset;
    const endIdx = endRunIdx + endRunLineOffset + endOffset;
    // console.log(startIdx, endIdx)
    if (!startIdx && !endIdx) return;
    const { runs } = this.state;
    const startRun = runs[startRunIdx];
    // console.log('Start Run', startRunIdx)
    const endRun = runs[endRunIdx];
    const newEndRun = { ...endRun, prev: startIdx };
    let i = startRun.end;
    while (i && i <= endRunIdx && runs[i]) {
      const l = runs[i].end;
      delete runs[i];
      i = l;
    }
    startRun.end = startIdx;
    if (!runs[endIdx]) {
      runs[endIdx] = newEndRun;
      if (runs[newEndRun.end]) {
        runs[newEndRun.end].prev = endIdx;
      }
    } else {
      runs[endIdx].prev = startIdx;
    }
    if (!runs[startIdx]) {
      runs[startIdx] = {
        type: name,
        end: endIdx,
        prev: startRunIdx,
      };
    } else {
      runs[startIdx].type = name;
      runs[startIdx].end = endIdx;
    }
    i = startIdx;
    // console.log('Merge start at', i)
    // Merge run before
    while (i && runs[i] && runs[i].prev != null) {
      const { prev } = runs[i];
      // console.log(prev)
      if (runs[prev].type === runs[i].type) {
        runs[prev].end = runs[i].end;
        delete runs[i];
        i = prev;
      } else break;
    }
    // Merge run after
    // console.log(i)
    while (runs[i]) {
      const next = runs[i].end;
      if (runs[next] && runs[next].type === runs[i].type) {
        runs[i].end = runs[next].end;
        delete runs[next];
      } else if (runs[next]) {
        runs[next].prev = i;
        break;
      } else {
        break;
      }
    }
    this.setState({ runs });
    this.props.onSaved(runs);
  };
  render() {
    const { text, runs } = this.state;
    const newLocal = this;
    let currentRuns = 0;
    return (
      <div className="text-area row">
        <div className="col-xs-12 col-sm-12 col-md-12 col-lg-12">
          {Object.keys(this.props.categories).map(this.createButton)}
          <button
            type="button"
            className="btn btn-default"
            key="Reset-btn"
            onClick={() => {
              newLocal.setState({
                runs: {
                  0: {
                    end: text.length,
                    type: 'normal',
                    prev: null,
                  },
                },
              });
            }}
          >
            Reset
          </button>
        </div>
        <div
          key="text-container"
          id={this.props.id}
          ref={function setContainer(container) {
            newLocal.container = container;
          }}
          className="text-container col-xs-12 col-sm-12 col-md-12 col-lg-12"
        >
          {Object.keys(runs).map((start) => {
            const { end, type } = runs[start];
            const { color } = this.props.categories[type];
            let len = 0;
            const temp = currentRuns;
            currentRuns = end;
            const parts = text.substring(start, end).split('\n');
            return parts.map((x, i) => {
              const id = len;
              len += x.length + 1;
              if (i < parts.length - 1) {
                return [
                  <span
                    key={`${temp}-${id}`}
                    id={`${temp}-${id}`}
                    title={type}
                    style={{
                      backgroundColor: color,
                      color: TinyColor(color).getBrightness() < 196 ? 'white' : 'black',
                    }}
                  >
                    {x}
                  </span>,
                  <br key={`${temp}br${id}`} />,
                ];
              }
              return (
                <span
                  title={type}
                  key={`${temp}-${id}`}
                  id={`${temp}-${id}`}
                  style={{
                    backgroundColor: color,
                    color: TinyColor(color).getBrightness() < 196 ? 'white' : 'black',
                  }}
                >
                  {x}
                </span>
              );
            });
          })}
        </div>
      </div>
    );
  }
}
