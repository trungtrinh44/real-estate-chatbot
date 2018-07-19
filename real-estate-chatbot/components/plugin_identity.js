module.exports = function(controller) {
  controller.middleware.spawn.use(function(bot, next) {
    if (controller.studio_identity) {
      bot.identity = controller.studio_identity;
      bot.identity.id = controller.studio_identity.botkit_studio_id;
    } else {
      bot.identity = {
          name: 'Botkit for Web',
          id: 'web',
      }
    }
    next();
  })
}
