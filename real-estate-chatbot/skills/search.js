const axios = require('axios');

const ASPECT_DISCOVERY_ENDPOINT = 'http://35.202.94.212:5000/api/v1/real-estate-extraction';
const SEARCH_ENDPOINT = 'http://35.232.164.158:4774/api/v1/posts';
const NUM_RESULTS = 5;
const MAX_RESULTS = 30;

const ASPECTS = {
  "addr_street": "Đường",
  "addr_district": "Quận",
  "addr_city": "Thành phố",
  "addr_ward": "Phường",
  "position": "Vị trí",
  "area": "Diện tích",
  "price": "Giá",
  "transaction_type": "Loại giao dịch",
  "realestate_type": "Loại BĐS",
  "legal": "Pháp lý",
  "potential": "Tiềm năng",
  "surrounding": "Khu vực",
  "surrounding_characteristics": "Đặc điểm khu vực",
  "surrounding_name": "Xung quanh",
  "interior_floor": "Số tầng",
  "interior_room": "Phòng",
  "orientation": "Hướng",
  "project": "Dự án"
}

module.exports = function (controller) {

  const history = {};

  function unhandledMessage(bot, message) {
    console.log(message)
    bot.startConversation(message, function (err, convo) {
      convo.say('Tôi không hiểu ý bạn lắm :)');
    });

  }

  function hello(bot, message) {
    bot.reply(message, "Xin chào, bạn cần tìm bất động sản như thế nào?")
  }

  function getCurrentAspects(data) {
    return data.filter(x => x.type != 'normal').map(item => {
      item.title = ASPECTS[item.type]
      return item
    })
  }

  async function check_aspects(text) {
    try {
      let result = await axios.post(
        ASPECT_DISCOVERY_ENDPOINT,
        JSON.stringify([text]),
        {
          headers: {
            'Content-Type': 'application/json'
          }
        });
      let tags = result.data[0].tags;
      return tags;
    } catch (error) {
      throw new Error(error)
    }
  }

  async function callAPI(data, isText = true, skip = 0) {
    let requestBody = {
      limit: MAX_RESULTS,
      skip: skip,
      string: isText,
    }
    if (isText)
      requestBody.query = data
    else
      requestBody.tags = data
    try {
      let results = await axios.post(SEARCH_ENDPOINT, JSON.stringify(requestBody), {
        headers: {
          'Content-Type': 'application/json'
        }
      })
      return results;
    } catch (error) {
      console.log(error);
      throw new Error(error);
    }
  }

  function hasUsefulAspects(tags) {
    let filtered = tags.filter(x => x.type !== 'normal')
    return filtered;
  }

  async function query(input, message, isText = true) {
    try {
      let results = await callAPI(input, isText)
      results = results.data;
      let queryData = results.data;
      let aspects = hasUsefulAspects(results.tags);
      if (aspects && aspects.length > 0) {
        let userId = message.user;
        history[userId] = {
          offset: NUM_RESULTS,
          data: queryData,
          query: aspects,
          text: isText ? message.text : '',
        };
        queryData = queryData.slice(0, NUM_RESULTS);
        return queryData;
      } else {
        // bot.reply(message, 'Bạn vui lòng cung cấp yêu cầu về bất động sản cần tìm')
      }
    } catch (error) {
      console.log(error);
      throw new Error(error)
    }
  }

  async function search(bot, message, isText = true, realMessageObject = null) {
    if (!realMessageObject) realMessageObject = message;
    try {
      let queryData = await query(isText ? message.text : message, message, isText)
      bot.startConversation(realMessageObject, function (err, convo) {
        if (queryData && queryData.length > 0) {
          convo.say({
            text: `Đây là ${NUM_RESULTS} kết quả tìm được theo yêu cầu của bạn`,
            articles: queryData
          })
          convo.say({
            text: 'Bạn có muốn xem thêm kết quả không',
            quick_replies: [
              {
                title: 'CÓ',
                payload: 'Xem thêm kết quả'
              },
              {
                title: 'KHÔNG',
                payload: 'Ngừng xem kết quả'
              },
            ]
          });
        }
        else {
          convo.say('Không tìm được kết quả nào với yêu cầu của bạn. Bạn vui lòng tìm kiếm với yêu cầu khác')
        }
      });
    } catch (error) {
      console.log(error);
      bot.reply(message, "Đã có lỗi xảy ra, bạn vui lòng thử lại với yêu cầu khác")
    }
  }


  controller.hears('Xem thêm kết quả', 'message_received', function (bot, message) {
    bot.startConversation(message, function (err, convo) {
      let user = message.user;
      let result;
      if (history[user] != undefined) {
        result = history[user].data
        //TODO check no more results
        result = result.slice(history[user].offset, history[user].offset + NUM_RESULTS)
        history[user].offset = history[user].offset + NUM_RESULTS;
      }
      if (result) {
        convo.say({
          text: `Đây là ${NUM_RESULTS} kết quả tiếp theo`,
          articles: result
        })
        convo.say({
          text: 'Bạn có muốn xem thêm kết quả không',
          quick_replies: [
            {
              title: 'CÓ',
              payload: 'Xem thêm kết quả'
            },
            {
              title: 'KHÔNG',
              payload: 'Ngừng xem kết quả'
            },
          ]
        });
      }
    });
  })

  controller.hears('Ngừng xem kết quả', 'message_received', function (bot, message) {
    bot.startConversation(message, function (err, convo) {
      convo.say({
        text: 'Bạn có muốn chỉnh sửa yêu cầu tìm kiếm không?',
        quick_replies: [
          {
            title: 'Thêm yêu cầu',
            payload: 'Thêm yêu cầu'
          },
          {
            title: 'Bỏ yêu cầu',
            payload: 'Bỏ yêu cầu'
          },
          {
            title: 'Tìm kiếm khác',
            payload: 'Tìm kiếm khác'
          },
          {
            title: 'Kết thúc',
            payload: 'Kết thúc'
          },
        ]
      });
    });
  })

  async function addAspects(oldAspect, aspects) {
    try {
      let tags = await check_aspects(aspects)
      let newQuery = oldAspect.concat(tags)
      return newQuery
    } catch (error) {
      throw new Error(error)
    }
  }

  controller.hears('Thêm yêu cầu', 'message_received', function (bot, message) {
    let user = message.user;
    if (history[user] == undefined) {
      bot.reply(message, "Bạn chưa có yêu cầu nào trước đó")
      return;
    }
    else {
      let oldQuery = history[user].query;
      let currentAspects = getCurrentAspects(oldQuery)
      bot.startConversation(message, function (err, convo) {
        convo.say({
          text: 'Đây là những yêu cầu hiện tại của bạn',
          aspects_list: currentAspects,
        });
        convo.addQuestion("Bạn cần thêm yêu cầu gì?", async (message, convo) => {
          try {
            let newQuery = await addAspects(currentAspects, message.text)
            newQuery = getCurrentAspects(newQuery)
            history[user].query = newQuery
            convo.say({
              text: 'Đây là những yêu cầu hiện tại của bạn',
              aspects_list: newQuery,

            });
            search(bot, newQuery, false, message)
          } catch (error) {
            // console.log(error)
            throw new Error(error)
          }
          convo.next();
        }, {}, 'default')
      });
    }
  })


  function removeAspect(oldQuery, aspects) {
    let newQuery = oldQuery.filter(x => !aspects.includes(x.content))
    return newQuery;
  }

  controller.hears('Bỏ yêu cầu: ', 'message_received', function (bot, message) {
    let user = message.user;
    if (history[user] == undefined) {
      bot.reply(message, "Bạn chưa có yêu cầu nào trước đó")
      return;
    }
    else {
      let oldMess = history[user];
      let toRemoveAspects = message.text.slice('Bỏ yêu cầu: '.length)
      toRemoveAspects = toRemoveAspects.split(',').map(x => x.trim());
      let result = removeAspect(oldMess.query, toRemoveAspects)
      history[user].query = result
      bot.startConversation(message, function (err, convo) {
        convo.say({
          text: 'Đây là những yêu cầu hiện tại của bạn',
          aspects_list: result,
        });
        search(bot, result, false, message)

      });
    }
  })

  controller.hears('Bỏ yêu cầu', 'message_received', function (bot, message) {
    let user = message.user;
    if (history[user] == undefined) {
      bot.reply(message, "Bạn chưa có yêu cầu nào trước đó")
      return;
    }
    else {
      let oldMess = history[user];
      let replies = getCurrentAspects(oldMess.query)
      bot.startConversation(message, function (err, convo) {
        convo.say({
          text: 'Đây là những yêu cầu hiện tại của bạn',
          multiple_replies: replies,
        });
      });
    }
  })

  controller.hears('Tìm kiếm khác', 'message_received', function (bot, message) {
    bot.reply(message, "Mời bạn nhập tìm kiếm khác")
  })

  controller.hears('Kết thúc', 'message_received', function (bot, message) {
    bot.startConversation(message, function (err, convo) {
      convo.say('Chúc bạn một ngày vui vẻ :)');
      convo.say('Nếu bạn có nhu cầu có thể tiếp tục hỏi');
    });
  })

  controller.on('hello', hello);
  controller.on('welcome_back', hello);
  controller.on('message_received', search);

}
