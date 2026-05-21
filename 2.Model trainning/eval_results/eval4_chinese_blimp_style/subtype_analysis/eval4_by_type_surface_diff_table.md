# Eval 4 By Type And Observed Surface Difference

Rows are grouped by broad phenomenon/type. The main subtype label is the observed good/bad Chinese surface difference, while UID is kept as a secondary identifier to avoid merging distinct official ZhoBLiMP paradigms.

## Loaded Data

- items: 35400
- score rows: 70800
- phenomena: 15
- models: chinese_4epoch, diacritic_matched_token_4epoch
- subtype rows: 3955

## BA

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad deletes 冯大哥; bad inserts 冯大哥 | BA_inversion | 4 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥把那六只老虎扔在派出所里。<br>Bad: 把那六只老虎冯大哥扔在派出所里。 |
| multiple edits: bad deletes 刘先生; bad inserts 刘先生 | BA_inversion | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生把那五头大象扔在火车站里。<br>Bad: 把那五头大象刘先生扔在火车站里。 |
| multiple edits: bad inserts 心脏被; bad deletes 把心脏 | BA_BEI_subj_drop | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们把心脏检查了，就走。<br>Bad: 心脏被你们检查了，就走。 |
| multiple edits: bad inserts 海洋被; bad deletes 把海洋 | BA_BEI_subj_drop | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们把海洋跨越了，就走。<br>Bad: 海洋被她们跨越了，就走。 |
| multiple edits: bad inserts 花卷被; bad deletes 把花卷 | BA_BEI_subj_drop | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们把花卷吃了，就来。<br>Bad: 花卷被他们吃了，就来。 |
| multiple edits: bad deletes 冯大哥的下属; bad inserts 冯大哥的下属 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥的下属把另外十只小猫藏在沙漠里。<br>Bad: 把另外十只小猫冯大哥的下属藏在沙漠里。 |
| multiple edits: bad deletes 另外四个儿子; bad inserts 另外四个儿子 | BA_inversion | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外四个儿子把另外六个老板藏在火山里。<br>Bad: 把另外六个老板另外四个儿子藏在火山里。 |
| multiple edits: bad deletes 周大妈; bad inserts 周大妈 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈把她们的司机藏在小桥里。<br>Bad: 把她们的司机周大妈藏在小桥里。 |
| multiple edits: bad deletes 她们的老板; bad inserts 她们的老板 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们的老板把这个打工人藏在照相馆里。<br>Bad: 把这个打工人她们的老板藏在照相馆里。 |
| multiple edits: bad deletes 她的妈妈; bad inserts 她的妈妈 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她的妈妈把何太太的记者扔在派出所里。<br>Bad: 把何太太的记者她的妈妈扔在派出所里。 |
| multiple edits: bad deletes 小王的老板; bad inserts 小王的老板 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小王的老板把另外三条鱼摆在火山里。<br>Bad: 把另外三条鱼小王的老板摆在火山里。 |
| multiple edits: bad deletes 张婶的姐姐; bad inserts 张婶的姐姐 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张婶的姐姐把那两位打工人扔在海洋里。<br>Bad: 把那两位打工人张婶的姐姐扔在海洋里。 |
| multiple edits: bad deletes 我们的妹妹; bad inserts 我们的妹妹 | BA_inversion | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们的妹妹把这个音乐家放在小桥里。<br>Bad: 把这个音乐家我们的妹妹放在小桥里。 |
| multiple edits: bad deletes 这一个司机; bad inserts 这一个司机 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这一个司机把另外九只鸡扔在派出所里。<br>Bad: 把另外九只鸡这一个司机扔在派出所里。 |
| multiple edits: bad deletes 这九个顾客; bad inserts 这九个顾客 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这九个顾客把我们的打工人藏在沙漠里。<br>Bad: 把我们的打工人这九个顾客藏在沙漠里。 |
| multiple edits: bad deletes 这位同事; bad inserts 这位同事 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位同事把王五的打工人摆在海洋里。<br>Bad: 把王五的打工人这位同事摆在海洋里。 |
| multiple edits: bad deletes 这位父亲; bad inserts 这位父亲 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位父亲把王姨的鱼藏在照相馆里。<br>Bad: 把王姨的鱼这位父亲藏在照相馆里。 |
| multiple edits: bad deletes 这位记者; bad inserts 这位记者 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位记者把王五的演奏员摆在山洞里。<br>Bad: 把王五的演奏员这位记者摆在山洞里。 |
| multiple edits: bad deletes 这八位员工; bad inserts 员工这八位 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这八位员工把王大娘的员工扔在俱乐部里。<br>Bad: 把王大娘的员工这八位员工扔在俱乐部里。 |
| multiple edits: bad deletes 那个吉他手; bad inserts 那个吉他手 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个吉他手把张三的弟弟放在照相馆里。<br>Bad: 把张三的弟弟那个吉他手放在照相馆里。 |
| multiple edits: bad deletes 那个工人; bad inserts 那个工人 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个工人把这十头牛藏在派出所里。<br>Bad: 把这十头牛那个工人藏在派出所里。 |
| multiple edits: bad deletes 那四个下属; bad inserts 那四个下属 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那四个下属把他们的姐姐藏在沙漠里。<br>Bad: 把他们的姐姐那四个下属藏在沙漠里。 |
| multiple edits: bad deletes 郑大妈的上级; bad inserts 郑大妈的上级 | BA_inversion | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 郑大妈的上级把徐小姐的罪犯摆在沙漠里。<br>Bad: 把徐小姐的罪犯郑大妈的上级摆在沙漠里。 |
| multiple edits: bad inserts 啤酒被; bad deletes 把啤酒 | BA_BEI_subj_drop | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们把啤酒买了，就去。<br>Bad: 啤酒被我们买了，就去。 |
| multiple edits: bad inserts 小猫被; bad deletes 把小猫 | BA_BEI_subj_drop | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们把小猫领养了，就走。<br>Bad: 小猫被我们领养了，就走。 |
| multiple edits: bad inserts 把他们的鸭; bad deletes 把他们的鸭 | BA_inversion | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小王的爸爸把他们的鸭摆在山洞里。<br>Bad: 把他们的鸭小王的爸爸摆在山洞里。 |
| multiple edits: bad inserts 把你的鸭; bad deletes 把你的鸭 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这五位学生把你的鸭摆在照相馆里。<br>Bad: 把你的鸭这五位学生摆在照相馆里。 |
| multiple edits: bad inserts 把另外一只鸭; bad deletes 把另外一只鸭 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 何太太的爸爸把另外一只鸭放在火车站里。<br>Bad: 把另外一只鸭何太太的爸爸放在火车站里。 |
| multiple edits: bad inserts 把另外七头牛; bad deletes 把另外七头牛 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥的朋友把另外七头牛扔在山洞里。<br>Bad: 把另外七头牛冯大哥的朋友扔在山洞里。 |
| multiple edits: bad inserts 把另外五条鱼; bad deletes 把另外五条鱼 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这五个打工人把另外五条鱼摆在海洋里。<br>Bad: 把另外五条鱼这五个打工人摆在海洋里。 |
| multiple edits: bad inserts 把她的父亲; bad deletes 把她的父亲 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位音乐家把她的父亲摆在沙漠里。<br>Bad: 把她的父亲那位音乐家摆在沙漠里。 |
| multiple edits: bad inserts 把她的鸭; bad deletes 把她的鸭 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那两个服务员把她的鸭扔在海洋里。<br>Bad: 把她的鸭那两个服务员扔在海洋里。 |
| multiple edits: bad inserts 把我们的鱼; bad deletes 把我们的鱼 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位演奏员把我们的鱼扔在海洋里。<br>Bad: 把我们的鱼那位演奏员扔在海洋里。 |
| multiple edits: bad inserts 把我们的鸭; bad deletes 把我们的鸭 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 郑大妈的兄弟把我们的鸭放在沙漠里。<br>Bad: 把我们的鸭郑大妈的兄弟放在沙漠里。 |
| multiple edits: bad inserts 把这个小孩; bad deletes 把这个小孩 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那四个音乐家把这个小孩摆在火山里。<br>Bad: 把这个小孩那四个音乐家摆在火山里。 |
| multiple edits: bad inserts 把这个弟弟; bad deletes 把这个弟弟 | BA_inversion | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这七位打工人把这个弟弟放在池塘里。<br>Bad: 把这个弟弟这七位打工人放在池塘里。 |
| multiple edits: bad inserts 把这位老师; bad deletes 把这位老师 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位服务员把这位老师放在沙漠里。<br>Bad: 把这位老师那位服务员放在沙漠里。 |
| multiple edits: bad inserts 把这八条蛇; bad deletes 把这八条蛇 | BA_inversion | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张三的儿子把这八条蛇扔在消防站里。<br>Bad: 把这八条蛇张三的儿子扔在消防站里。 |
| multiple edits: bad inserts 把那七条鱼; bad deletes 把那七条鱼 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷的兄弟把那七条鱼藏在沙漠里。<br>Bad: 把那七条鱼胡大爷的兄弟藏在沙漠里。 |
| multiple edits: bad inserts 把那个上级; bad deletes 上级把那个 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外四位上级把那个上级放在海洋里。<br>Bad: 把那个上级另外四位上级放在海洋里。 |
| multiple edits: bad inserts 把那九位学生; bad deletes 把那九位学生 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈的姐姐把那九位学生扔在派出所里。<br>Bad: 把那九位学生周大妈的姐姐扔在派出所里。 |
| multiple edits: bad inserts 把那八条蛇; bad deletes 把那八条蛇 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小王的姐妹把那八条蛇放在火山里。<br>Bad: 把那八条蛇小王的姐妹放在火山里。 |
| multiple edits: bad inserts 把那四只鸭; bad deletes 把那四只鸭 | BA_inversion | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外六位空姐把那四只鸭摆在派出所里。<br>Bad: 把那四只鸭另外六位空姐摆在派出所里。 |
| multiple edits: bad inserts 歌曲被; bad deletes 把歌曲 | BA_BEI_subj_drop | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们把歌曲演奏了，就走。<br>Bad: 歌曲被她们演奏了，就走。 |
| multiple edits: bad inserts 胃被; bad deletes 把胃 | BA_BEI_subj_drop | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我把胃检查了，就来。<br>Bad: 胃被我检查了，就来。 |
| multiple edits: bad inserts 蛋糕被; bad deletes 把蛋糕 | BA_BEI_subj_drop | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们把蛋糕吃了，就走。<br>Bad: 蛋糕被我们吃了，就走。 |
| multiple edits: bad inserts 香蕉被; bad deletes 把香蕉 | BA_BEI_subj_drop | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们把香蕉吃了，就来。<br>Bad: 香蕉被我们吃了，就来。 |
| multiple edits: 在预习 -> 把; bad inserts 预习 | BA_no_progressive | 12 | 0.9167 | 0.0000 | +0.9167 | 0.0000 | Good: 张夫人正在预习那本教材。<br>Bad: 张夫人正把那本教材预习。 |
| multiple edits: bad deletes 徐小姐; bad inserts 徐小姐 | BA_inversion | 6 | 0.8333 | 0.0000 | +0.8333 | 0.0000 | Good: 徐小姐把我们的哥哥摆在山洞里。<br>Bad: 把我们的哥哥徐小姐摆在山洞里。 |
| multiple edits: bad inserts 鼻子被; bad deletes 把鼻子 | BA_BEI_subj_drop | 5 | 0.2000 | 1.0000 | -0.8000 | 0.0000 | Good: 他们把鼻子包扎了，就去。<br>Bad: 鼻子被他们包扎了，就去。 |
| multiple edits: bad deletes 王姨; bad inserts 王姨 | BA_inversion | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 王姨把这两位打工人放在火山里。<br>Bad: 把这两位打工人王姨放在火山里。 |
| multiple edits: bad inserts 大象被; bad deletes 把大象 | BA_BEI_subj_drop | 4 | 0.2500 | 1.0000 | -0.7500 | 0.0000 | Good: 你们把大象麻醉了，就来。<br>Bad: 大象被你们麻醉了，就来。 |
| multiple edits: bad deletes 胡大爷; bad inserts 胡大爷 | BA_inversion | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 胡大爷把我们的工人藏在俱乐部里。<br>Bad: 把我们的工人胡大爷藏在俱乐部里。 |
| multiple edits: bad inserts 把那头大象; bad deletes 把那头大象 | BA_inversion | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 这八个记者把那头大象藏在俱乐部里。<br>Bad: 把那头大象这八个记者藏在俱乐部里。 |
| multiple edits: 在看 -> 把; bad inserts 看 | BA_no_progressive | 17 | 0.8235 | 0.1765 | +0.6471 | 0.0000 | Good: 我们正在看这部小说。<br>Bad: 我们正把这部小说看。 |
| multiple edits: bad deletes 他把; bad inserts 被他 | BA_BEI_subj_drop | 23 | 0.3913 | 1.0000 | -0.6087 | 0.0000 | Good: 他把电影拍摄了，就去。<br>Bad: 电影被他拍摄了，就去。 |
| multiple edits: bad deletes 你们把; bad inserts 被你们 | BA_BEI_subj_drop | 10 | 0.3000 | 0.9000 | -0.6000 | 0.0000 | Good: 你们把方便面吃了，就走。<br>Bad: 方便面被你们吃了，就走。 |
| multiple edits: bad inserts 把鸭; bad deletes 把鸭 | BA_negation | 14 | 1.0000 | 0.4286 | +0.5714 | 0.0000 | Good: 胡大爷没有把鸭捕捉。<br>Bad: 胡大爷把鸭没有捕捉。 |
| multiple edits: bad deletes 我们把; bad inserts 被我们 | BA_BEI_subj_drop | 7 | 0.2857 | 0.8571 | -0.5714 | 0.0000 | Good: 我们把录像带看了，就走。<br>Bad: 录像带被我们看了，就走。 |
| multiple edits: 在捕捉 -> 把; bad inserts 捕捉 | BA_no_progressive | 10 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 她正在捕捉那条蛇。<br>Bad: 她正把那条蛇捕捉。 |
| multiple edits: bad deletes 她; bad inserts 她 | BA_inversion | 8 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 她把这位顾客扔在火车站里。<br>Bad: 把这位顾客她扔在火车站里。 |
| multiple edits: bad deletes 张婶; bad inserts 张婶 | BA_inversion | 6 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 张婶把她们的大象藏在电影院里。<br>Bad: 把她们的大象张婶藏在电影院里。 |
| multiple edits: bad inserts 书被; bad deletes 把书 | BA_BEI_subj_drop | 6 | 0.6667 | 0.1667 | +0.5000 | 0.0000 | Good: 他们把书看了，就走。<br>Bad: 书被他们看了，就走。 |
| multiple edits: bad deletes 张三; bad inserts 张三 | BA_inversion | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 张三把那四位钢琴家扔在沙漠里。<br>Bad: 把那四位钢琴家张三扔在沙漠里。 |
| multiple edits: bad inserts 被子被; bad deletes 把被子 | BA_BEI_subj_drop | 4 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 他们把被子盖了，就走。<br>Bad: 被子被他们盖了，就走。 |
| multiple edits: bad deletes 小明; bad inserts 小明 | BA_inversion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 小明把那位消费者扔在派出所里。<br>Bad: 把那位消费者小明扔在派出所里。 |
| multiple edits: bad deletes 小王; bad inserts 小王 | BA_inversion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 小王把陈大姐的姐妹扔在沙漠里。<br>Bad: 把陈大姐的姐妹小王扔在沙漠里。 |
| multiple edits: bad deletes 李先生; bad inserts 李先生 | BA_inversion | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 李先生把杨大哥的老虎放在火车站里。<br>Bad: 把杨大哥的老虎李先生放在火车站里。 |
| multiple edits: bad deletes 这四位演员; bad inserts 这四位演员 | BA_inversion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这四位演员把这位打工人摆在火车站里。<br>Bad: 把这位打工人这四位演员摆在火车站里。 |
| multiple edits: bad deletes 那个舞者; bad inserts 那个舞者 | BA_inversion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那个舞者把吴太太的鸡放在沙漠里。<br>Bad: 把吴太太的鸡那个舞者放在沙漠里。 |
| multiple edits: bad deletes 那位母亲; bad inserts 那位母亲 | BA_inversion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那位母亲把张婶的演员放在火车站里。<br>Bad: 把张婶的演员那位母亲放在火车站里。 |
| multiple edits: bad deletes 郑大妈; bad inserts 郑大妈 | BA_inversion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 郑大妈把他们的鸡扔在海洋里。<br>Bad: 把他们的鸡郑大妈扔在海洋里。 |
| multiple edits: bad inserts 卡车被; bad deletes 把卡车 | BA_BEI_subj_drop | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他们把卡车开了，就走。<br>Bad: 卡车被他们开了，就走。 |
| multiple edits: bad inserts 小说被; bad deletes 把小说 | BA_BEI_subj_drop | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 你们把小说写了，就走。<br>Bad: 小说被你们写了，就走。 |
| multiple edits: bad inserts 戏曲被; bad deletes 把戏曲 | BA_BEI_subj_drop | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 她们把戏曲唱了，就去。<br>Bad: 戏曲被她们唱了，就去。 |
| multiple edits: bad inserts 把头; bad deletes 把头 | BA_negation | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 宋女士没有把头包扎。<br>Bad: 宋女士把头没有包扎。 |
| multiple edits: bad inserts 把歌; bad deletes 把歌 | BA_negation | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 我们没有把歌唱。<br>Bad: 我们把歌没有唱。 |
| multiple edits: bad inserts 视频被; bad deletes 把视频 | BA_BEI_subj_drop | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 你们把视频拍摄了，就走。<br>Bad: 视频被你们拍摄了，就走。 |
| multiple edits: 在吃 -> 把; bad inserts 吃 | BA_no_progressive | 16 | 0.6250 | 0.1875 | +0.4375 | 0.0000 | Good: 张夫人正在吃这片面包。<br>Bad: 张夫人正把这片面包吃。 |
| multiple edits: 在拍摄 -> 把; bad inserts 拍摄 | BA_no_progressive | 12 | 1.0000 | 0.5833 | +0.4167 | 0.0000 | Good: 张婶正在拍摄这部电影。<br>Bad: 张婶正把这部电影拍摄。 |
| multiple edits: bad deletes 我把; bad inserts 被我 | BA_BEI_subj_drop | 29 | 0.5172 | 0.9310 | -0.4138 | 0.0000 | Good: 我把小提琴拉了，就去。<br>Bad: 小提琴被我拉了，就去。 |
| bad inserts 所 | BA_suo_adverbial_b | 300 | 0.8633 | 0.4500 | +0.4133 | 0.0000 | Good: 你把这些裤子寄完了。<br>Bad: 你把这些裤子所寄完了。 |
| multiple edits: bad deletes 吴太太; bad inserts 吴太太 | BA_inversion | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 吴太太把另外五头大象扔在火车站里。<br>Bad: 把另外五头大象吴太太扔在火车站里。 |
| multiple edits: 在弹 -> 把; bad inserts 弹 | BA_no_progressive | 8 | 0.3750 | 0.0000 | +0.3750 | 0.0000 | Good: 她们正在弹这个玻璃珠。<br>Bad: 她们正把这个玻璃珠弹。 |
| 恨 → 怕 | BA_no_stative_verb | 300 | 0.8633 | 0.4900 | +0.3733 | 0.0000 | Good: 何太太会把何太太恨一整天。<br>Bad: 何太太会把何太太怕一整天。 |
| multiple edits: 在喝 -> 把; bad inserts 喝 | BA_no_progressive | 14 | 0.9286 | 0.5714 | +0.3571 | 0.0000 | Good: 他正在喝那瓶红酒。<br>Bad: 他正把那瓶红酒喝。 |
| multiple edits: bad inserts 脚被; bad deletes 把脚 | BA_BEI_subj_drop | 6 | 0.5000 | 0.8333 | -0.3333 | 0.0000 | Good: 他们把脚打断了，就去。<br>Bad: 脚被他们打断了，就去。 |
| multiple edits: bad deletes 他们; bad inserts 他们 | BA_inversion | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 他们把王姨的记者放在电影院里。<br>Bad: 把王姨的记者他们放在电影院里。 |
| multiple edits: bad deletes 王大娘; bad inserts 王大娘 | BA_inversion | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 王大娘把宋女士的母亲放在派出所里。<br>Bad: 把宋女士的母亲王大娘放在派出所里。 |
| multiple edits: bad inserts 把蛇; bad deletes 把蛇 | BA_negation | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 陈大姐没有把蛇捕捉。<br>Bad: 陈大姐把蛇没有捕捉。 |
| multiple edits: bad inserts 轮船被; bad deletes 把轮船 | BA_BEI_subj_drop | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 你们把轮船驾驶了，就走。<br>Bad: 轮船被你们驾驶了，就走。 |
| multiple edits: 在清洗 -> 把; bad inserts 清洗 | BA_no_progressive | 15 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 赵大爷正在清洗那个杯子。<br>Bad: 赵大爷正把那个杯子清洗。 |
| multiple edits: bad inserts 老虎被; bad deletes 把老虎 | BA_BEI_subj_drop | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 她们把老虎麻醉了，就来。<br>Bad: 老虎被她们麻醉了，就来。 |
| multiple edits: bad inserts 腿被; bad deletes 把腿 | BA_BEI_subj_drop | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 她把腿打断了，就来。<br>Bad: 腿被她打断了，就来。 |
| multiple edits: 在检查 -> 把; bad inserts 检查 | BA_no_progressive | 12 | 0.9167 | 0.5833 | +0.3333 | 0.0000 | Good: 我们正在检查那只脚。<br>Bad: 我们正把那只脚检查。 |
| multiple edits: 在制作 -> 把; bad inserts 制作 | BA_no_progressive | 13 | 1.0000 | 0.6923 | +0.3077 | 0.0000 | Good: 她们正在制作那本手账。<br>Bad: 她们正把那本手账制作。 |
| multiple edits: bad deletes 把卡车; bad inserts 卡车 | BA_deletion | 20 | 1.0000 | 0.7000 | +0.3000 | 0.0000 | Good: 徐小姐把卡车放满了椅子。<br>Bad: 徐小姐放满了卡车椅子。 |
| multiple edits: bad deletes 李四; bad inserts 李四 | BA_inversion | 7 | 0.7143 | 0.4286 | +0.2857 | 0.0000 | Good: 李四把这条鱼藏在山洞里。<br>Bad: 把这条鱼李四藏在山洞里。 |
| multiple edits: 在打断 -> 把; bad inserts 打断 | BA_no_progressive | 14 | 0.0000 | 0.2857 | -0.2857 | 0.0000 | Good: 何太太正在打断那只脚。<br>Bad: 何太太正把那只脚打断。 |
| multiple edits: bad deletes 没有; bad inserts 没有 | BA_negation | 214 | 0.9112 | 0.6495 | +0.2617 | 0.0000 | Good: 王姨没有把沙漠跨越。<br>Bad: 王姨把沙漠没有跨越。 |
| multiple edits: bad inserts 把鸡; bad deletes 把鸡 | BA_negation | 24 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 小王没有把鸡清蒸。<br>Bad: 小王把鸡没有清蒸。 |
| multiple edits: 在创作 -> 把; bad inserts 创作 | BA_no_progressive | 20 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 小明正在创作那部漫画。<br>Bad: 小明正把那部漫画创作。 |
| multiple edits: bad inserts 鸭被; bad deletes 把鸭 | BA_BEI_subj_drop | 16 | 0.6250 | 0.8750 | -0.2500 | 0.0000 | Good: 我把鸭煮了，就去。<br>Bad: 鸭被我煮了，就去。 |
| multiple edits: bad inserts 教材被; bad deletes 把教材 | BA_BEI_subj_drop | 8 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 他们把教材预习了，就走。<br>Bad: 教材被他们预习了，就走。 |
| multiple edits: 在清蒸 -> 把; bad inserts 清蒸 | BA_no_progressive | 8 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 张夫人正在清蒸那只鸭。<br>Bad: 张夫人正把那只鸭清蒸。 |
| multiple edits: bad inserts 把手; bad deletes 把手 | BA_negation | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 你没有把手包扎。<br>Bad: 你把手没有包扎。 |
| multiple edits: bad inserts 把腿; bad deletes 把腿 | BA_negation | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 张婶没有把腿包扎。<br>Bad: 张婶把腿没有包扎。 |
| multiple edits: bad inserts 糖被; bad deletes 把糖 | BA_BEI_subj_drop | 4 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 他把糖买了，就去。<br>Bad: 糖被他买了，就去。 |
| multiple edits: bad deletes 她把; bad inserts 被她 | BA_BEI_subj_drop | 29 | 0.7586 | 1.0000 | -0.2414 | 0.0000 | Good: 她把卡车开了，就走。<br>Bad: 卡车被她开了，就走。 |
| multiple edits: 在观看 -> 把; bad inserts 观看 | BA_no_progressive | 18 | 0.9444 | 0.7222 | +0.2222 | 0.0000 | Good: 杨大哥正在观看那部电影。<br>Bad: 杨大哥正把那部电影观看。 |
| 让 → 把 | causative_shi_ba | 155 | 0.9419 | 0.7226 | +0.2194 | 0.0000 | Good: 周大妈让小明比较苦恼。<br>Bad: 周大妈把小明比较苦恼。 |
| multiple edits: bad deletes 她们把; bad inserts 被她们 | BA_BEI_subj_drop | 5 | 0.6000 | 0.8000 | -0.2000 | 0.0000 | Good: 她们把协奏曲演奏了，就走。<br>Bad: 协奏曲被她们演奏了，就走。 |
| multiple edits: bad inserts 电影被; bad deletes 把电影 | BA_BEI_subj_drop | 5 | 0.6000 | 0.8000 | -0.2000 | 0.0000 | Good: 我们把电影观看了，就来。<br>Bad: 电影被我们观看了，就来。 |
| multiple edits: 在爆炒 -> 把; bad inserts 爆炒 | BA_no_progressive | 10 | 0.2000 | 0.0000 | +0.2000 | 0.0000 | Good: 她正在爆炒这条鱼。<br>Bad: 她正把这条鱼爆炒。 |
| multiple edits: bad deletes 把火车; bad inserts 火车 | BA_deletion | 30 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 王先生把火车摆满了被子。<br>Bad: 王先生摆满了火车被子。 |
| multiple edits: bad inserts 鱼被; bad deletes 把鱼 | BA_BEI_subj_drop | 20 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 他把鱼炖了，就来。<br>Bad: 鱼被他炖了，就来。 |
| multiple edits: bad deletes 赵大爷; bad inserts 赵大爷 | BA_inversion | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 赵大爷把另外五头大象放在照相馆里。<br>Bad: 把另外五头大象赵大爷放在照相馆里。 |
| multiple edits: bad deletes 把货箱; bad inserts 货箱 | BA_deletion | 154 | 0.9545 | 0.7727 | +0.1818 | 0.0000 | Good: 何太太把货箱摆满了饮料瓶。<br>Bad: 何太太摆满了货箱饮料瓶。 |
| multiple edits: bad deletes 杨大哥; bad inserts 杨大哥 | BA_inversion | 6 | 0.8333 | 0.6667 | +0.1667 | 0.0000 | Good: 杨大哥把李太太的鸭放在池塘里。<br>Bad: 把李太太的鸭杨大哥放在池塘里。 |
| multiple edits: 在包扎 -> 把; bad inserts 包扎 | BA_no_progressive | 12 | 0.1667 | 0.0000 | +0.1667 | 0.0000 | Good: 你正在包扎那条腿。<br>Bad: 你正把那条腿包扎。 |
| multiple edits: bad inserts 把牛; bad deletes 把牛 | BA_negation | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 我们没有把牛屠宰。<br>Bad: 我们把牛没有屠宰。 |
| 使 → 把 | causative_shi_ba | 145 | 0.4966 | 0.3517 | +0.1448 | 0.0000 | Good: 赵大爷使李四非常开心。<br>Bad: 赵大爷把李四非常开心。 |
| multiple edits: bad deletes 我; bad inserts 我 | BA_inversion | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 我把他的姐妹摆在照相馆里。<br>Bad: 把他的姐妹我摆在照相馆里。 |
| multiple edits: 在麻醉 -> 把; bad inserts 麻醉 | BA_no_progressive | 14 | 0.1429 | 0.0000 | +0.1429 | 0.0000 | Good: 他们正在麻醉那只老虎。<br>Bad: 他们正把那只老虎麻醉。 |
| multiple edits: 在炖 -> 把; bad inserts 炖 | BA_no_progressive | 15 | 0.1333 | 0.0000 | +0.1333 | 0.0000 | Good: 她们正在炖这条鱼。<br>Bad: 她们正把这条鱼炖。 |
| bad inserts 了 | BA_verb_le_a | 300 | 0.5900 | 0.7200 | -0.1300 | 0.0000 | Good: 张三把王五夸奖了。<br>Bad: 张三把了王五夸奖了。 |
| multiple edits: 在领养 -> 把; bad inserts 领养 | BA_no_progressive | 16 | 0.5000 | 0.3750 | +0.1250 | 0.0000 | Good: 他们正在领养这条小狗。<br>Bad: 他们正把这条小狗领养。 |
| multiple edits: bad deletes 他; bad inserts 他 | BA_inversion | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 他把王大娘的老板藏在山洞里。<br>Bad: 把王大娘的老板他藏在山洞里。 |
| multiple edits: bad inserts 牛被; bad deletes 把牛 | BA_BEI_subj_drop | 8 | 0.6250 | 0.7500 | -0.1250 | 0.0000 | Good: 我把牛屠宰了，就走。<br>Bad: 牛被我屠宰了，就走。 |
| multiple edits: bad inserts 鸡被; bad deletes 把鸡 | BA_BEI_subj_drop | 17 | 0.7647 | 0.8824 | -0.1176 | 0.0000 | Good: 我们把鸡爆炒了，就走。<br>Bad: 鸡被我们爆炒了，就走。 |
| bad inserts 把没 | BA_meiba | 300 | 0.5467 | 0.4500 | +0.0967 | 0.0000 | Good: 郑大妈把赵大爷教育了。<br>Bad: 郑大妈把没把赵大爷教育了。 |
| multiple edits: bad deletes 你把; bad inserts 被你 | BA_BEI_subj_drop | 21 | 0.8095 | 0.9048 | -0.0952 | 0.0000 | Good: 你把小猫领养了，就来。<br>Bad: 小猫被你领养了，就来。 |
| multiple edits: bad deletes 她们; bad inserts 她们 | BA_inversion | 12 | 1.0000 | 0.9167 | +0.0833 | 0.0000 | Good: 她们把这只鸡藏在派出所里。<br>Bad: 把这只鸡她们藏在派出所里。 |
| bad deletes 了 | BA_verb_le_b | 300 | 0.9833 | 0.9100 | +0.0733 | 0.0000 | Good: 胡大爷把我批评了。<br>Bad: 胡大爷把我批评。 |
| multiple edits: bad deletes 把货车; bad inserts 货车 | BA_deletion | 29 | 0.8966 | 0.9655 | -0.0690 | 0.0000 | Good: 你把货车放满了收音机。<br>Bad: 你放满了货车收音机。 |
| multiple edits: bad inserts 把鱼; bad deletes 把鱼 | BA_negation | 24 | 0.9583 | 0.9167 | +0.0417 | 0.0000 | Good: 小王没有把鱼爆炒。<br>Bad: 小王把鱼没有爆炒。 |
| bad inserts 所 | BA_suo_adverbial_a | 300 | 0.2733 | 0.2600 | +0.0133 | 0.0000 | Good: 我们把冯大哥诽谤得非常苦恼。<br>Bad: 我们把冯大哥所诽谤得非常苦恼。 |
| bad inserts 她 | BA_duplicate_argument | 167 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥把宋女士夸奖了一下。<br>Bad: 杨大哥把宋女士夸奖了她一下。 |
| bad inserts 他 | BA_duplicate_argument | 133 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐把刘先生嘉奖了一下。<br>Bad: 王小姐把刘先生嘉奖了他一下。 |
| multiple edits: bad deletes 把轮船; bad inserts 轮船 | BA_deletion | 37 | 0.9459 | 0.9459 | +0.0000 | 0.0000 | Good: 他们把轮船放满了饮料瓶。<br>Bad: 他们放满了轮船饮料瓶。 |
| multiple edits: bad deletes 把飞机; bad inserts 飞机 | BA_deletion | 30 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们把飞机摆满了手套。<br>Bad: 她们摆满了飞机手套。 |
| multiple edits: bad deletes 你; bad inserts 你 | BA_inversion | 17 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你把这六条鱼放在山洞里。<br>Bad: 把这六条鱼你放在山洞里。 |
| multiple edits: 在烧 -> 把; bad inserts 烧 | BA_no_progressive | 14 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太正在烧那只鸭。<br>Bad: 吴太太正把那只鸭烧。 |
| multiple edits: 在屠宰 -> 把; bad inserts 屠宰 | BA_no_progressive | 12 | 0.3333 | 0.3333 | +0.0000 | 0.0000 | Good: 他正在屠宰那头牛。<br>Bad: 他正把那头牛屠宰。 |
| multiple edits: bad deletes 你们; bad inserts 你们 | BA_inversion | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们把我的服务员摆在海洋里。<br>Bad: 把我的服务员你们摆在海洋里。 |
| multiple edits: 在煮 -> 把; bad inserts 煮 | BA_no_progressive | 9 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥正在煮那条鱼。<br>Bad: 冯大哥正把那条鱼煮。 |
| multiple edits: 在盖 -> 把; bad inserts 盖 | BA_no_progressive | 9 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你正在盖这条被子。<br>Bad: 你正把这条被子盖。 |
| multiple edits: bad deletes 我们; bad inserts 我们 | BA_inversion | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把那条鱼摆在照相馆里。<br>Bad: 把那条鱼我们摆在照相馆里。 |
| multiple edits: bad inserts 杯子被; bad deletes 把杯子 | BA_BEI_subj_drop | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把杯子清洗了，就走。<br>Bad: 杯子被我们清洗了，就走。 |
| multiple edits: bad deletes 他们把; bad inserts 被他们 | BA_BEI_subj_drop | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 他们把小提琴拉了，就来。<br>Bad: 小提琴被他们拉了，就来。 |
| multiple edits: bad inserts 把这条鱼; bad deletes 把这条鱼 | BA_inversion | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八个顾客把这条鱼扔在沙漠里。<br>Bad: 把这条鱼这八个顾客扔在沙漠里。 |
| multiple edits: bad deletes 李太太; bad inserts 李太太 | BA_inversion | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太把这位演员摆在池塘里。<br>Bad: 把这位演员李太太摆在池塘里。 |
| multiple edits: bad inserts 把这头牛; bad deletes 把这头牛 | BA_inversion | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那十位钢琴家把这头牛摆在山洞里。<br>Bad: 把这头牛那十位钢琴家摆在山洞里。 |
| multiple edits: bad deletes 张夫人; bad inserts 张夫人 | BA_inversion | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人把冯大哥的老虎藏在山洞里。<br>Bad: 把冯大哥的老虎张夫人藏在山洞里。 |
| multiple edits: bad deletes 王小姐; bad inserts 王小姐 | BA_inversion | 4 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 王小姐把这五位顾客藏在派出所里。<br>Bad: 把这五位顾客王小姐藏在派出所里。 |
| multiple edits: bad deletes 张先生; bad inserts 张先生 | BA_inversion | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生把那八头牛藏在火山里。<br>Bad: 把那八头牛张先生藏在火山里。 |
| multiple edits: bad inserts 手账被; bad deletes 把手账 | BA_BEI_subj_drop | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们把手账制作了，就去。<br>Bad: 手账被你们制作了，就去。 |
| multiple edits: bad deletes 何太太; bad inserts 何太太 | BA_inversion | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太把宋女士的司机藏在池塘里。<br>Bad: 把宋女士的司机何太太藏在池塘里。 |
| multiple edits: bad deletes 王五; bad inserts 王五 | BA_inversion | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五把这头大象摆在消防站里。<br>Bad: 把这头大象王五摆在消防站里。 |
| multiple edits: bad deletes 王先生; bad inserts 王先生 | BA_inversion | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生把这六头大象放在山洞里。<br>Bad: 把这六头大象王先生放在山洞里。 |
| multiple edits: bad deletes 那个上级; bad inserts 那个上级 | BA_inversion | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个上级把那位领导放在火车站里。<br>Bad: 把那位领导那个上级放在火车站里。 |
| multiple edits: bad inserts 古筝被; bad deletes 把古筝 | BA_BEI_subj_drop | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们把古筝弹了，就走。<br>Bad: 古筝被他们弹了，就走。 |
| multiple edits: bad inserts 手被; bad deletes 把手 | BA_BEI_subj_drop | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你把手包扎了，就去。<br>Bad: 手被你包扎了，就去。 |
| multiple edits: bad inserts 把我们的鸡; bad deletes 把我们的鸡 | BA_inversion | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的父亲把我们的鸡扔在海洋里。<br>Bad: 把我们的鸡他们的父亲扔在海洋里。 |
| multiple edits: bad inserts 把脚; bad deletes 把脚 | BA_negation | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太没有把脚打断。<br>Bad: 吴太太把脚没有打断。 |
| multiple edits: bad inserts 把那只鸭; bad deletes 把那只鸭 | BA_inversion | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外六位同事把那只鸭放在派出所里。<br>Bad: 把那只鸭另外六位同事放在派出所里。 |
| multiple edits: bad inserts 椅子被; bad deletes 把椅子 | BA_BEI_subj_drop | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把椅子搬了，就去。<br>Bad: 椅子被我们搬了，就去。 |
| multiple edits: bad inserts 沙漠被; bad deletes 把沙漠 | BA_BEI_subj_drop | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们把沙漠跨越了，就走。<br>Bad: 沙漠被她们跨越了，就走。 |
| multiple edits: bad inserts 火车被; bad deletes 把火车 | BA_BEI_subj_drop | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们把火车驾驶了，就去。<br>Bad: 火车被他们驾驶了，就去。 |
| multiple edits: bad inserts 眼睛被; bad deletes 把眼睛 | BA_BEI_subj_drop | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 你们把眼睛检查了，就去。<br>Bad: 眼睛被你们检查了，就去。 |
| multiple edits: bad inserts 红茶被; bad deletes 把红茶 | BA_BEI_subj_drop | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们把红茶喝了，就走。<br>Bad: 红茶被你们喝了，就走。 |
| multiple edits: bad deletes 他们的朋友; bad inserts 他们的朋友 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的朋友把王小姐的演员扔在沙漠里。<br>Bad: 把王小姐的演员他们的朋友扔在沙漠里。 |
| multiple edits: bad deletes 他的同事; bad inserts 他的同事 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的同事把这头大象扔在俱乐部里。<br>Bad: 把这头大象他的同事扔在俱乐部里。 |
| multiple edits: bad deletes 他的妈妈; bad inserts 他的妈妈 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的妈妈把那十条鱼藏在电影院里。<br>Bad: 把那十条鱼他的妈妈藏在电影院里。 |
| multiple edits: bad deletes 他的领导; bad inserts 他的领导 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的领导把何太太的鱼扔在沙漠里。<br>Bad: 把何太太的鱼他的领导扔在沙漠里。 |
| multiple edits: bad deletes 你们的儿子; bad inserts 你们的儿子 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的儿子把那七只老虎藏在小桥里。<br>Bad: 把那七只老虎你们的儿子藏在小桥里。 |
| multiple edits: bad deletes 你们的员工; bad inserts 你们的员工 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的员工把我们的上级藏在池塘里。<br>Bad: 把我们的上级你们的员工藏在池塘里。 |
| multiple edits: bad deletes 你们的老板; bad inserts 你们的老板 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的老板把刘先生的领导放在派出所里。<br>Bad: 把刘先生的领导你们的老板放在派出所里。 |
| multiple edits: bad deletes 你的老师; bad inserts 你的老师 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的老师把这四只鸡放在山洞里。<br>Bad: 把这四只鸡你的老师放在山洞里。 |
| multiple edits: bad deletes 另外七个小孩; bad inserts 另外七个小孩 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外七个小孩把那四位服务员摆在消防站里。<br>Bad: 把那四位服务员另外七个小孩摆在消防站里。 |
| multiple edits: bad deletes 宋女士; bad inserts 宋女士 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士把另外七只鸭摆在海洋里。<br>Bad: 把另外七只鸭宋女士摆在海洋里。 |
| multiple edits: bad deletes 小明的同事; bad inserts 小明的同事 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明的同事把另外八个妹妹摆在照相馆里。<br>Bad: 把另外八个妹妹小明的同事摆在照相馆里。 |
| multiple edits: bad deletes 张婶的员工; bad inserts 员工张婶的 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶的员工把那八位员工藏在俱乐部里。<br>Bad: 把那八位员工张婶的员工藏在俱乐部里。 |
| multiple edits: bad deletes 我们的弟弟; bad inserts 我们的弟弟 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的弟弟把那四头大象扔在电影院里。<br>Bad: 把那四头大象我们的弟弟扔在电影院里。 |
| multiple edits: bad deletes 我们的领导; bad inserts 我们的领导 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的领导把另外三只老虎放在俱乐部里。<br>Bad: 把另外三只老虎我们的领导放在俱乐部里。 |
| multiple edits: bad deletes 我的同事; bad inserts 我的同事 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的同事把我们的吉他手摆在池塘里。<br>Bad: 把我们的吉他手我的同事摆在池塘里。 |
| multiple edits: bad deletes 王五的上级; bad inserts 王五的上级 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五的上级把那八位钢琴家藏在消防站里。<br>Bad: 把那八位钢琴家王五的上级藏在消防站里。 |
| multiple edits: bad deletes 这七位空姐; bad inserts 这七位空姐 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七位空姐把他们的司机藏在池塘里。<br>Bad: 把他们的司机这七位空姐藏在池塘里。 |
| multiple edits: bad deletes 这个下属; bad inserts 这个下属 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个下属把那四头牛放在电影院里。<br>Bad: 把那四头牛这个下属放在电影院里。 |
| multiple edits: bad deletes 这个司机; bad inserts 这个司机 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个司机把宋女士的鱼摆在俱乐部里。<br>Bad: 把宋女士的鱼这个司机摆在俱乐部里。 |
| multiple edits: bad deletes 这个哥哥; bad inserts 这个哥哥 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个哥哥把这五条鱼放在海洋里。<br>Bad: 把这五条鱼这个哥哥放在海洋里。 |
| multiple edits: bad deletes 这个消费者; bad inserts 这个消费者 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个消费者把王大娘的鱼摆在派出所里。<br>Bad: 把王大娘的鱼这个消费者摆在派出所里。 |
| multiple edits: bad deletes 这位司机; bad inserts 这位司机 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位司机把我们的鸭藏在海洋里。<br>Bad: 把我们的鸭这位司机藏在海洋里。 |
| multiple edits: bad deletes 这位演员; bad inserts 这位演员 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位演员把那一只鸭放在派出所里。<br>Bad: 把那一只鸭这位演员放在派出所里。 |
| multiple edits: bad deletes 这位音乐家; bad inserts 这位音乐家 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位音乐家把张夫人的蛇扔在小桥里。<br>Bad: 把张夫人的蛇这位音乐家扔在小桥里。 |
| multiple edits: bad deletes 这十个弟弟; bad inserts 这十个弟弟 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这十个弟弟把那六个记者扔在火山里。<br>Bad: 把那六个记者这十个弟弟扔在火山里。 |
| multiple edits: bad deletes 这十位同事; bad inserts 这十位同事 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这十位同事把这两位工人放在山洞里。<br>Bad: 把这两位工人这十位同事放在山洞里。 |
| multiple edits: bad deletes 那七个领导; bad inserts 那七个领导 | BA_inversion | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那七个领导把另外十条鱼摆在派出所里。<br>Bad: 把另外十条鱼那七个领导摆在派出所里。 |
| multiple edits: bad deletes 那三位司机; bad inserts 那三位司机 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那三位司机把那两头大象扔在俱乐部里。<br>Bad: 把那两头大象那三位司机扔在俱乐部里。 |
| multiple edits: bad deletes 那三位工人; bad inserts 那三位工人 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那三位工人把这五头大象放在海洋里。<br>Bad: 把这五头大象那三位工人放在海洋里。 |
| multiple edits: bad deletes 那两位顾客; bad inserts 那两位顾客 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那两位顾客把另外三头牛扔在消防站里。<br>Bad: 把另外三头牛那两位顾客扔在消防站里。 |
| multiple edits: bad deletes 那个儿子; bad inserts 那个儿子 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个儿子把那两条鱼藏在山洞里。<br>Bad: 把那两条鱼那个儿子藏在山洞里。 |
| multiple edits: bad deletes 那个妹妹; bad inserts 那个妹妹 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个妹妹把另外九条蛇扔在海洋里。<br>Bad: 把另外九条蛇那个妹妹扔在海洋里。 |
| multiple edits: bad deletes 那个姐姐; bad inserts 那个姐姐 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个姐姐把这六只鸡扔在消防站里。<br>Bad: 把这六只鸡那个姐姐扔在消防站里。 |
| multiple edits: bad deletes 那个朋友; bad inserts 那个朋友 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个朋友把王先生的领导扔在山洞里。<br>Bad: 把王先生的领导那个朋友扔在山洞里。 |
| multiple edits: bad deletes 那个演员; bad inserts 那个演员 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个演员把这只老虎放在池塘里。<br>Bad: 把这只老虎那个演员放在池塘里。 |
| multiple edits: bad deletes 那九个小孩; bad inserts 那九个小孩 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九个小孩把李太太的老虎扔在海洋里。<br>Bad: 把李太太的老虎那九个小孩扔在海洋里。 |
| multiple edits: bad deletes 那位上级; bad inserts 那位上级 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位上级把张三的鸭放在消防站里。<br>Bad: 把张三的鸭那位上级放在消防站里。 |
| multiple edits: bad deletes 那位学生; bad inserts 那位学生 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位学生把那五条小狗放在山洞里。<br>Bad: 把那五条小狗那位学生放在山洞里。 |
| multiple edits: bad deletes 那位消费者; bad inserts 那位消费者 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位消费者把杨大哥的爸爸放在火车站里。<br>Bad: 把杨大哥的爸爸那位消费者放在火车站里。 |
| multiple edits: bad deletes 那位演员; bad inserts 那位演员 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位演员把那四个打工人放在沙漠里。<br>Bad: 把那四个打工人那位演员放在沙漠里。 |
| multiple edits: bad deletes 那位演奏员; bad inserts 那位演奏员 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位演奏员把宋女士的打工人摆在沙漠里。<br>Bad: 把宋女士的打工人那位演奏员摆在沙漠里。 |
| multiple edits: bad deletes 那位父亲; bad inserts 那位父亲 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位父亲把吴太太的鸭扔在沙漠里。<br>Bad: 把吴太太的鸭那位父亲扔在沙漠里。 |
| multiple edits: bad deletes 那位老师; bad inserts 那位老师 | BA_inversion | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位老师把她的演奏员放在消防站里。<br>Bad: 把她的演奏员那位老师放在消防站里。 |
| multiple edits: bad deletes 那八个音乐家; bad inserts 那八个音乐家 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那八个音乐家把那两个演奏员藏在电影院里。<br>Bad: 把那两个演奏员那八个音乐家藏在电影院里。 |
| multiple edits: bad deletes 那八位同事; bad inserts 那八位同事 | BA_inversion | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那八位同事把另外七个老板放在照相馆里。<br>Bad: 把另外七个老板那八位同事放在照相馆里。 |
| multiple edits: bad deletes 那八位空姐; bad inserts 那八位空姐 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那八位空姐把张三的工人放在池塘里。<br>Bad: 把张三的工人那八位空姐放在池塘里。 |
| multiple edits: bad deletes 那十位上级; bad inserts 那十位上级 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那十位上级把这三位吉他手摆在池塘里。<br>Bad: 把这三位吉他手那十位上级摆在池塘里。 |
| multiple edits: bad deletes 陈大姐; bad inserts 陈大姐 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐把那条蛇藏在小桥里。<br>Bad: 把那条蛇陈大姐藏在小桥里。 |
| multiple edits: bad inserts 京剧被; bad deletes 把京剧 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们把京剧唱了，就走。<br>Bad: 京剧被她们唱了，就走。 |
| multiple edits: bad inserts 咖啡被; bad deletes 把咖啡 | BA_BEI_subj_drop | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们把咖啡喝了，就来。<br>Bad: 咖啡被她们喝了，就来。 |
| multiple edits: bad inserts 坚果被; bad deletes 把坚果 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把坚果吃了，就去。<br>Bad: 坚果被我们吃了，就去。 |
| multiple edits: bad inserts 头被; bad deletes 把头 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她把头检查了，就来。<br>Bad: 头被她检查了，就来。 |
| multiple edits: bad inserts 小狗被; bad deletes 把小狗 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们把小狗领养了，就去。<br>Bad: 小狗被他们领养了，就去。 |
| multiple edits: bad inserts 把书; bad deletes 把书 | BA_negation | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没有把书看。<br>Bad: 他们把书没有看。 |
| multiple edits: bad inserts 把冯大哥的鸡; bad deletes 把冯大哥的鸡 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘的母亲把冯大哥的鸡扔在海洋里。<br>Bad: 把冯大哥的鸡王大娘的母亲扔在海洋里。 |
| multiple edits: bad inserts 把刘先生的顾客; bad deletes 把刘先生的顾客 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外两位吉他手把刘先生的顾客摆在沙漠里。<br>Bad: 把刘先生的顾客另外两位吉他手摆在沙漠里。 |
| multiple edits: bad inserts 把另外十头牛; bad deletes 把另外十头牛 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九位音乐家把另外十头牛藏在山洞里。<br>Bad: 把另外十头牛这九位音乐家藏在山洞里。 |
| multiple edits: bad inserts 把吴太太的鸭; bad deletes 把吴太太的鸭 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一位吉他手把吴太太的鸭藏在小桥里。<br>Bad: 把吴太太的鸭另外一位吉他手藏在小桥里。 |
| multiple edits: bad inserts 把她们的牛; bad deletes 把她们的牛 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘的下属把她们的牛摆在山洞里。<br>Bad: 把她们的牛王大娘的下属摆在山洞里。 |
| multiple edits: bad inserts 把她们的鱼; bad deletes 把她们的鱼 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个消费者把她们的鱼放在俱乐部里。<br>Bad: 把她们的鱼那个消费者放在俱乐部里。 |
| multiple edits: bad inserts 把她的蛇; bad deletes 把她的蛇 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十位母亲把她的蛇扔在电影院里。<br>Bad: 把她的蛇另外十位母亲扔在电影院里。 |
| multiple edits: bad inserts 把徐小姐的罪犯; bad deletes 把徐小姐的罪犯 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外两个钢琴家把徐小姐的罪犯放在火车站里。<br>Bad: 把徐小姐的罪犯另外两个钢琴家放在火车站里。 |
| multiple edits: bad inserts 把我的姐妹; bad deletes 把我的姐妹 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的妈妈把我的姐妹藏在池塘里。<br>Bad: 把我的姐妹她们的妈妈藏在池塘里。 |
| multiple edits: bad inserts 把我的演奏员; bad deletes 把我的演奏员 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外六个钢琴家把我的演奏员摆在照相馆里。<br>Bad: 把我的演奏员另外六个钢琴家摆在照相馆里。 |
| multiple edits: bad inserts 把这七条蛇; bad deletes 把这七条蛇 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶的下属把这七条蛇藏在电影院里。<br>Bad: 把这七条蛇张婶的下属藏在电影院里。 |
| multiple edits: bad inserts 把这两条鱼; bad deletes 把这两条鱼 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外九个舞者把这两条鱼藏在小桥里。<br>Bad: 把这两条鱼另外九个舞者藏在小桥里。 |
| multiple edits: bad inserts 把这九头大象; bad deletes 把这九头大象 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘的下属把这九头大象摆在海洋里。<br>Bad: 把这九头大象王大娘的下属摆在海洋里。 |
| multiple edits: bad inserts 把这九头牛; bad deletes 把这九头牛 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的领导把这九头牛摆在沙漠里。<br>Bad: 把这九头牛你们的领导摆在沙漠里。 |
| multiple edits: bad inserts 把这位司机; bad deletes 把这位司机 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太的领导把这位司机藏在山洞里。<br>Bad: 把这位司机李太太的领导藏在山洞里。 |
| multiple edits: bad inserts 把这八只鸭; bad deletes 把这八只鸭 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一个姐姐把这八只鸭放在山洞里。<br>Bad: 把这八只鸭另外一个姐姐放在山洞里。 |
| multiple edits: bad inserts 把这八条鱼; bad deletes 把这八条鱼 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这两个姐姐把这八条鱼藏在沙漠里。<br>Bad: 把这八条鱼这两个姐姐藏在沙漠里。 |
| multiple edits: bad inserts 把这十只鸭; bad deletes 把这十只鸭 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九位音乐家把这十只鸭藏在电影院里。<br>Bad: 把这十只鸭这九位音乐家藏在电影院里。 |
| multiple edits: bad inserts 把这只鸡; bad deletes 把这只鸡 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明的朋友把这只鸡藏在小桥里。<br>Bad: 把这只鸡小明的朋友藏在小桥里。 |
| multiple edits: bad inserts 把这只鸭; bad deletes 把这只鸭 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一位上级把这只鸭扔在海洋里。<br>Bad: 把这只鸭另外一位上级扔在海洋里。 |
| multiple edits: bad inserts 把这头大象; bad deletes 把这头大象 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那两个工人把这头大象扔在火山里。<br>Bad: 把这头大象那两个工人扔在火山里。 |
| multiple edits: bad inserts 把那三条鱼; bad deletes 把那三条鱼 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位消费者把那三条鱼放在海洋里。<br>Bad: 把那三条鱼这位消费者放在海洋里。 |
| multiple edits: bad inserts 把那位下属; bad deletes 把那位下属 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位演奏员把那位下属放在电影院里。<br>Bad: 把那位下属这位演奏员放在电影院里。 |
| multiple edits: bad inserts 把那位吉他手; bad deletes 把那位吉他手 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三位消费者把那位吉他手摆在山洞里。<br>Bad: 把那位吉他手这三位消费者摆在山洞里。 |
| multiple edits: bad inserts 把那位工人; bad deletes 把那位工人 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外五位顾客把那位工人放在火车站里。<br>Bad: 把那位工人另外五位顾客放在火车站里。 |
| multiple edits: bad inserts 把那八位服务员; bad deletes 把那八位服务员 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外三位钢琴家把那八位服务员摆在小桥里。<br>Bad: 把那八位服务员另外三位钢琴家摆在小桥里。 |
| multiple edits: bad inserts 把那十个司机; bad deletes 把那十个司机 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十个小孩把那十个司机摆在小桥里。<br>Bad: 把那十个司机另外十个小孩摆在小桥里。 |
| multiple edits: bad inserts 把那只鸡; bad deletes 把那只鸡 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的妹妹把那只鸡藏在沙漠里。<br>Bad: 把那只鸡他们的妹妹藏在沙漠里。 |
| multiple edits: bad inserts 把那四只鸡; bad deletes 把那四只鸡 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位打工人把那四只鸡藏在电影院里。<br>Bad: 把那四只鸡这位打工人藏在电影院里。 |
| multiple edits: bad inserts 把那四头牛; bad deletes 把那四头牛 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外四个罪犯把那四头牛藏在消防站里。<br>Bad: 把那四头牛另外四个罪犯藏在消防站里。 |
| multiple edits: bad inserts 把那头牛; bad deletes 把那头牛 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的上级把那头牛扔在沙漠里。<br>Bad: 把那头牛他们的上级扔在沙漠里。 |
| multiple edits: bad inserts 把那条鱼; bad deletes 把那条鱼 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个奴隶把那条鱼放在山洞里。<br>Bad: 把那条鱼那个奴隶放在山洞里。 |
| multiple edits: bad inserts 日记被; bad deletes 把日记 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们把日记写了，就走。<br>Bad: 日记被他们写了，就走。 |
| multiple edits: bad inserts 桌子被; bad deletes 把桌子 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们把桌子搬了，就去。<br>Bad: 桌子被她们搬了，就去。 |
| multiple edits: bad inserts 漫画被; bad deletes 把漫画 | BA_BEI_subj_drop | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们把漫画看了，就走。<br>Bad: 漫画被他们看了，就走。 |
| multiple edits: bad inserts 白酒被; bad deletes 把白酒 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把白酒喝了，就来。<br>Bad: 白酒被我们喝了，就来。 |
| multiple edits: bad inserts 肚子被; bad deletes 把肚子 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们把肚子包扎了，就去。<br>Bad: 肚子被她们包扎了，就去。 |
| multiple edits: bad inserts 蛇被; bad deletes 把蛇 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们把蛇捕捉了，就走。<br>Bad: 蛇被她们捕捉了，就走。 |
| multiple edits: bad inserts 货车被; bad deletes 把货车 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把货车开了，就来。<br>Bad: 货车被我们开了，就来。 |
| multiple edits: bad inserts 馒头被; bad deletes 把馒头 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们把馒头吃了，就走。<br>Bad: 馒头被他们吃了，就走。 |
| multiple edits: bad inserts 鱼丸被; bad deletes 把鱼丸 | BA_BEI_subj_drop | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们把鱼丸买了，就走。<br>Bad: 鱼丸被我们买了，就走。 |
| multiple edits: 这十 -> 把另外一; 把另外一 -> 这十 | BA_inversion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这十位服务员把另外一位服务员放在沙漠里。<br>Bad: 把另外一位服务员这十位服务员放在沙漠里。 |

## anaphor

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| 赵大爷 → 李太太 | anaphor_gender_agreement | 5 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷比较尊重他自己。<br>Bad: 李太太比较尊重他自己。 |
| 胡大爷 → 何太太 | anaphor_gender_agreement | 4 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷嫌弃他自己。<br>Bad: 何太太嫌弃他自己。 |
| multiple edits: 赵 -> 王; 爷 -> 娘 | anaphor_gender_agreement | 3 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷有点憎恨他自己。<br>Bad: 王大娘有点憎恨他自己。 |
| 刘先生 → 郑大妈 | anaphor_gender_agreement | 3 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生喜欢他自己。<br>Bad: 郑大妈喜欢他自己。 |
| 李太太 → 赵大爷 | anaphor_gender_agreement | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李太太批评了她自己。<br>Bad: 赵大爷批评了她自己。 |
| multiple edits: 周 -> 冯; 妈 -> 哥 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈很憎恨她自己。<br>Bad: 冯大哥很憎恨她自己。 |
| multiple edits: 杨 -> 陈; 哥 -> 姐 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥很憎恨他自己。<br>Bad: 陈大姐很憎恨他自己。 |
| multiple edits: 胡 -> 郑; 爷 -> 妈 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷鼓励了他自己。<br>Bad: 郑大妈鼓励了他自己。 |
| 何太太 → 杨大哥 | anaphor_gender_agreement | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太厌恶她自己。<br>Bad: 杨大哥厌恶她自己。 |
| 刘先生 → 吴太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生称赞了他自己。<br>Bad: 吴太太称赞了他自己。 |
| 刘先生 → 徐小姐 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生反感他自己。<br>Bad: 徐小姐反感他自己。 |
| 吴太太 → 冯大哥 | anaphor_gender_agreement | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 吴太太约束了她自己。<br>Bad: 冯大哥约束了她自己。 |
| 张先生 → 何太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张先生批评了他自己。<br>Bad: 何太太批评了他自己。 |
| 张先生 → 徐小姐 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张先生称赞了他自己。<br>Bad: 徐小姐称赞了他自己。 |
| 张先生 → 郑大妈 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张先生非常厌恶他自己。<br>Bad: 郑大妈非常厌恶他自己。 |
| 徐小姐 → 冯大哥 | anaphor_gender_agreement | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐憎恨她自己。<br>Bad: 冯大哥憎恨她自己。 |
| 徐小姐 → 刘先生 | anaphor_gender_agreement | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐欺骗了她自己。<br>Bad: 刘先生欺骗了她自己。 |
| 李先生 → 何太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生照顾了他自己。<br>Bad: 何太太照顾了他自己。 |
| 李先生 → 吴太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生诽谤了他自己。<br>Bad: 吴太太诽谤了他自己。 |
| 李先生 → 陈大姐 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生比较支持他自己。<br>Bad: 陈大姐比较支持他自己。 |
| 杨大哥 → 李太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥嘉奖了他自己。<br>Bad: 李太太嘉奖了他自己。 |
| 王先生 → 周大妈 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生责备了他自己。<br>Bad: 周大妈责备了他自己。 |
| 王先生 → 李太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生最反感他自己。<br>Bad: 李太太最反感他自己。 |
| 赵大爷 → 何太太 | anaphor_gender_agreement | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷提醒了他自己。<br>Bad: 何太太提醒了他自己。 |
| multiple edits: 冯 -> 王; 哥 -> 娘 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥比较支持他自己。<br>Bad: 王大娘比较支持他自己。 |
| multiple edits: 冯 -> 陈; 哥 -> 姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥提醒了他自己。<br>Bad: 陈大姐提醒了他自己。 |
| multiple edits: 杨 -> 王; 哥 -> 娘 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥最喜欢他自己。<br>Bad: 王大娘最喜欢他自己。 |
| multiple edits: 胡 -> 陈; 爷 -> 姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷有点尊重他自己。<br>Bad: 陈大姐有点尊重他自己。 |
| multiple edits: 陈 -> 杨; 姐 -> 哥 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 陈大姐支持她自己。<br>Bad: 杨大哥支持她自己。 |
| 何太太 → 冯大哥 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太欺骗了她自己。<br>Bad: 冯大哥欺骗了她自己。 |
| 先生 → 太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生鼓励了他自己。<br>Bad: 李太太鼓励了他自己。 |
| 冯大哥 → 何太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥表扬了他自己。<br>Bad: 何太太表扬了他自己。 |
| 冯大哥 → 王小姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥有点埋怨他自己。<br>Bad: 王小姐有点埋怨他自己。 |
| 刘先生 → 李太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生欺骗了他自己。<br>Bad: 李太太欺骗了他自己。 |
| 吴太太 → 杨大哥 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 吴太太憎恨她自己。<br>Bad: 杨大哥憎恨她自己。 |
| 吴太太 → 王先生 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 吴太太嘉奖了她自己。<br>Bad: 王先生嘉奖了她自己。 |
| 吴太太 → 胡大爷 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 吴太太有点厌恶她自己。<br>Bad: 胡大爷有点厌恶她自己。 |
| 周大妈 → 李先生 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈很赞赏她自己。<br>Bad: 李先生很赞赏她自己。 |
| 周大妈 → 王先生 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈比较支持她自己。<br>Bad: 王先生比较支持她自己。 |
| 夫人 → 先生 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张夫人非常埋怨她自己。<br>Bad: 张先生非常埋怨她自己。 |
| 宋女士 → 王先生 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 宋女士更喜欢她自己。<br>Bad: 王先生更喜欢她自己。 |
| 小姐 → 先生 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王小姐欺骗了她自己。<br>Bad: 王先生欺骗了她自己。 |
| 张先生 → 陈大姐 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张先生有点支持他自己。<br>Bad: 陈大姐有点支持他自己。 |
| 徐小姐 → 胡大爷 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐比较埋怨她自己。<br>Bad: 胡大爷比较埋怨她自己。 |
| 李先生 → 徐小姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生更反感他自己。<br>Bad: 徐小姐更反感他自己。 |
| 李太太 → 杨大哥 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李太太比较埋怨她自己。<br>Bad: 杨大哥比较埋怨她自己。 |
| 杨大哥 → 何太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥伤害了他自己。<br>Bad: 何太太伤害了他自己。 |
| 王先生 → 吴太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生批评了他自己。<br>Bad: 吴太太批评了他自己。 |
| 王先生 → 徐小姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生鼓励了他自己。<br>Bad: 徐小姐鼓励了他自己。 |
| 王先生 → 郑大妈 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生最支持他自己。<br>Bad: 郑大妈最支持他自己。 |
| 胡大爷 → 徐小姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷欺骗了他自己。<br>Bad: 徐小姐欺骗了他自己。 |
| 胡大爷 → 李太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷奖励了他自己。<br>Bad: 李太太奖励了他自己。 |
| 赵大爷 → 吴太太 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷很厌恶他自己。<br>Bad: 吴太太很厌恶他自己。 |
| 赵大爷 → 徐小姐 | anaphor_gender_agreement | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷尊重他自己。<br>Bad: 徐小姐尊重他自己。 |
| 郑大妈 → 李先生 | anaphor_gender_agreement | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 郑大妈最憎恨她自己。<br>Bad: 李先生最憎恨她自己。 |
| 她 → 他 | principle_A_domain | 148 | 0.9392 | 0.0000 | +0.9392 | 1.0000 | Good: 小王觉得张婶埋怨她自己。<br>Bad: 小王觉得张婶埋怨他自己。 |
| 她 → 他 | principle_A_c_command | 160 | 0.8812 | 0.0000 | +0.8812 | 1.0000 | Good: 冯大哥的母亲更憎恨她自己。<br>Bad: 冯大哥的母亲更憎恨他自己。 |
| 徐小姐 → 赵大爷 | anaphor_gender_agreement | 5 | 0.0000 | 0.8000 | -0.8000 | 0.0000 | Good: 徐小姐喜欢她自己。<br>Bad: 赵大爷喜欢她自己。 |
| multiple edits: 周 -> 杨; 妈 -> 哥 | anaphor_gender_agreement | 4 | 0.7500 | 0.0000 | +0.7500 | 0.0000 | Good: 周大妈憎恨她自己。<br>Bad: 杨大哥憎恨她自己。 |
| 吴太太 → 李先生 | anaphor_gender_agreement | 4 | 0.2500 | 1.0000 | -0.7500 | 0.0000 | Good: 吴太太支持她自己。<br>Bad: 李先生支持她自己。 |
| 杨大哥 → 徐小姐 | anaphor_gender_agreement | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 杨大哥安慰了他自己。<br>Bad: 徐小姐安慰了他自己。 |
| 刘先生 → 张夫人 | anaphor_gender_agreement | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 刘先生厌恶他自己。<br>Bad: 张夫人厌恶他自己。 |
| 吴太太 → 赵大爷 | anaphor_gender_agreement | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 吴太太反感她自己。<br>Bad: 赵大爷反感她自己。 |
| 徐小姐 → 张先生 | anaphor_gender_agreement | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 徐小姐支持她自己。<br>Bad: 张先生支持她自己。 |
| 杨大哥 → 吴太太 | anaphor_gender_agreement | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 杨大哥支持他自己。<br>Bad: 吴太太支持他自己。 |
| 徐小姐 → 李先生 | anaphor_gender_agreement | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 徐小姐诽谤了她自己。<br>Bad: 李先生诽谤了她自己。 |
| 王先生 → 陈大姐 | anaphor_gender_agreement | 3 | 0.6667 | 0.0000 | +0.6667 | 0.0000 | Good: 王先生夸奖了他自己。<br>Bad: 陈大姐夸奖了他自己。 |
| 宋女士 → 刘先生 | anaphor_gender_agreement | 5 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 宋女士反感她自己。<br>Bad: 刘先生反感她自己。 |
| 宋女士 → 张先生 | anaphor_gender_agreement | 4 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 宋女士原谅了她自己。<br>Bad: 张先生原谅了她自己。 |
| multiple edits: 杨 -> 郑; 哥 -> 妈 | anaphor_gender_agreement | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 杨大哥憎恨他自己。<br>Bad: 郑大妈憎恨他自己。 |
| multiple edits: 胡 -> 周; 爷 -> 妈 | anaphor_gender_agreement | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 胡大爷喜欢他自己。<br>Bad: 周大妈喜欢他自己。 |
| multiple edits: 胡 -> 王; 爷 -> 娘 | anaphor_gender_agreement | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 胡大爷厌恶他自己。<br>Bad: 王大娘厌恶他自己。 |
| 何太太 → 李先生 | anaphor_gender_agreement | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 何太太鼓励了她自己。<br>Bad: 李先生鼓励了她自己。 |
| 何太太 → 王先生 | anaphor_gender_agreement | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 何太太表扬了她自己。<br>Bad: 王先生表扬了她自己。 |
| 周大妈 → 刘先生 | anaphor_gender_agreement | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 周大妈非常嫌弃她自己。<br>Bad: 刘先生非常嫌弃她自己。 |
| 太太 → 先生 | anaphor_gender_agreement | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 李太太伤害了她自己。<br>Bad: 李先生伤害了她自己。 |
| 宋女士 → 胡大爷 | anaphor_gender_agreement | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 宋女士表扬了她自己。<br>Bad: 胡大爷表扬了她自己。 |
| 张先生 → 周大妈 | anaphor_gender_agreement | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 张先生奖励了他自己。<br>Bad: 周大妈奖励了他自己。 |
| 徐小姐 → 杨大哥 | anaphor_gender_agreement | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 徐小姐有点赞赏她自己。<br>Bad: 杨大哥有点赞赏她自己。 |
| 王小姐 → 张先生 | anaphor_gender_agreement | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 王小姐有点憎恨她自己。<br>Bad: 张先生有点憎恨她自己。 |
| bad inserts 们 | principle_A_domain_number | 300 | 0.4800 | 0.8567 | -0.3767 | 0.0000 | Good: 这好几百位服务员说李太太表扬了她自己。<br>Bad: 这好几百位服务员说李太太表扬了她们自己。 |
| multiple edits: 王 -> 胡; 娘 -> 爷 | anaphor_gender_agreement | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 王大娘厌恶她自己。<br>Bad: 胡大爷厌恶她自己。 |
| 何太太 → 张先生 | anaphor_gender_agreement | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 何太太伤害了她自己。<br>Bad: 张先生伤害了她自己。 |
| 冯大哥 → 李太太 | anaphor_gender_agreement | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 冯大哥批评了他自己。<br>Bad: 李太太批评了他自己。 |
| 陈大姐 → 王先生 | anaphor_gender_agreement | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 陈大姐最憎恨她自己。<br>Bad: 王先生最憎恨她自己。 |
| 张先生 → 李太太 | anaphor_gender_agreement | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 张先生反感他自己。<br>Bad: 李太太反感他自己。 |
| 张夫人 → 胡大爷 | anaphor_gender_agreement | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 张夫人最反感她自己。<br>Bad: 胡大爷最反感她自己。 |
| 王小姐 → 杨大哥 | anaphor_gender_agreement | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 王小姐有点厌恶她自己。<br>Bad: 杨大哥有点厌恶她自己。 |
| bad deletes 们 | principle_A_c_command_number | 300 | 0.5967 | 0.3433 | +0.2533 | 0.0000 | Good: 王先生的兄弟们嘉奖了他们自己。<br>Bad: 王先生的兄弟们嘉奖了他自己。 |
| multiple edits: 郑 -> 杨; 妈 -> 哥 | anaphor_gender_agreement | 4 | 0.2500 | 0.5000 | -0.2500 | 0.0000 | Good: 郑大妈很厌恶她自己。<br>Bad: 杨大哥很厌恶她自己。 |
| multiple edits: 郑 -> 胡; 妈 -> 爷 | anaphor_gender_agreement | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 郑大妈嘉奖了她自己。<br>Bad: 胡大爷嘉奖了她自己。 |
| 刘先生 → 王小姐 | anaphor_gender_agreement | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 刘先生支持他自己。<br>Bad: 王小姐支持他自己。 |
| 王小姐 → 胡大爷 | anaphor_gender_agreement | 4 | 0.2500 | 0.0000 | +0.2500 | 0.0000 | Good: 王小姐非常支持她自己。<br>Bad: 胡大爷非常支持她自己。 |
| bad deletes 们 | anaphor_number_agreement | 300 | 0.5267 | 0.3533 | +0.1733 | 0.0000 | Good: 那五个女儿原谅了她们自己。<br>Bad: 那五个女儿原谅了她自己。 |
| 他 → 她 | principle_A_domain | 152 | 0.1118 | 0.0000 | +0.1118 | 1.0000 | Good: 她认为赵大爷批评了他自己。<br>Bad: 她认为赵大爷批评了她自己。 |
| 他 → 她 | principle_A_c_command | 140 | 0.0929 | 0.0000 | +0.0929 | 1.0000 | Good: 王五的弟弟埋怨他自己。<br>Bad: 王五的弟弟埋怨她自己。 |
| 王大娘 → 刘先生 | anaphor_gender_agreement | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘鼓励了她自己。<br>Bad: 刘先生鼓励了她自己。 |
| 胡大爷 → 张夫人 | anaphor_gender_agreement | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷更厌恶他自己。<br>Bad: 张夫人更厌恶他自己。 |
| 大娘 → 先生 | anaphor_gender_agreement | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘反感她自己。<br>Bad: 王先生反感她自己。 |
| 张夫人 → 王先生 | anaphor_gender_agreement | 4 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 张夫人最厌恶她自己。<br>Bad: 王先生最厌恶她自己。 |
| 李先生 → 王小姐 | anaphor_gender_agreement | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生比较赞赏他自己。<br>Bad: 王小姐比较赞赏他自己。 |
| 李太太 → 胡大爷 | anaphor_gender_agreement | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太最嫌弃她自己。<br>Bad: 胡大爷最嫌弃她自己。 |
| 冯大哥 → 宋女士 | anaphor_gender_agreement | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥埋怨他自己。<br>Bad: 宋女士埋怨他自己。 |
| 宋女士 → 杨大哥 | anaphor_gender_agreement | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士支持她自己。<br>Bad: 杨大哥支持她自己。 |
| 宋女士 → 赵大爷 | anaphor_gender_agreement | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士嫌弃她自己。<br>Bad: 赵大爷嫌弃她自己。 |
| 张先生 → 王小姐 | anaphor_gender_agreement | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生厌恶他自己。<br>Bad: 王小姐厌恶他自己。 |
| 张夫人 → 杨大哥 | anaphor_gender_agreement | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人反感她自己。<br>Bad: 杨大哥反感她自己。 |
| 杨大哥 → 张夫人 | anaphor_gender_agreement | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥非常喜欢他自己。<br>Bad: 张夫人非常喜欢他自己。 |
| 胡大爷 → 宋女士 | anaphor_gender_agreement | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷反感他自己。<br>Bad: 宋女士反感他自己。 |
| multiple edits: 王 -> 赵; 娘 -> 爷 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘更喜欢她自己。<br>Bad: 赵大爷更喜欢她自己。 |
| multiple edits: 陈 -> 胡; 姐 -> 爷 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐很支持她自己。<br>Bad: 胡大爷很支持她自己。 |
| 何太太 → 胡大爷 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太反感她自己。<br>Bad: 胡大爷反感她自己。 |
| 先生 → 小姐 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生表扬了他自己。<br>Bad: 王小姐表扬了他自己。 |
| 刘先生 → 宋女士 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生约束了他自己。<br>Bad: 宋女士约束了他自己。 |
| 宋女士 → 冯大哥 | anaphor_gender_agreement | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士称赞了她自己。<br>Bad: 冯大哥称赞了她自己。 |
| 张先生 → 宋女士 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生嫌弃他自己。<br>Bad: 宋女士嫌弃他自己。 |
| 李先生 → 周大妈 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生厌恶他自己。<br>Bad: 周大妈厌恶他自己。 |
| 李太太 → 刘先生 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太教育了她自己。<br>Bad: 刘先生教育了她自己。 |
| 王先生 → 张夫人 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生比较嫌弃他自己。<br>Bad: 张夫人比较嫌弃他自己。 |
| 王小姐 → 刘先生 | anaphor_gender_agreement | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 王小姐原谅了她自己。<br>Bad: 刘先生原谅了她自己。 |
| 王小姐 → 赵大爷 | anaphor_gender_agreement | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 王小姐支持她自己。<br>Bad: 赵大爷支持她自己。 |
| 胡大爷 → 王小姐 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷埋怨他自己。<br>Bad: 王小姐埋怨他自己。 |
| 赵大爷 → 张夫人 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷比较埋怨他自己。<br>Bad: 张夫人比较埋怨他自己。 |
| 郑大妈 → 王先生 | anaphor_gender_agreement | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈批评了她自己。<br>Bad: 王先生批评了她自己。 |
| multiple edits: 冯 -> 郑; 哥 -> 妈 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥很尊重他自己。<br>Bad: 郑大妈很尊重他自己。 |
| multiple edits: 王 -> 杨; 娘 -> 哥 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘批评了她自己。<br>Bad: 杨大哥批评了她自己。 |
| multiple edits: 赵 -> 周; 爷 -> 妈 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷比较厌恶他自己。<br>Bad: 周大妈比较厌恶他自己。 |
| multiple edits: 赵 -> 陈; 爷 -> 姐 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷支持他自己。<br>Bad: 陈大姐支持他自己。 |
| multiple edits: 郑 -> 冯; 妈 -> 哥 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈夸奖了她自己。<br>Bad: 冯大哥夸奖了她自己。 |
| multiple edits: 陈 -> 冯; 姐 -> 哥 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐有点喜欢她自己。<br>Bad: 冯大哥有点喜欢她自己。 |
| multiple edits: 陈 -> 赵; 姐 -> 爷 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐诽谤了她自己。<br>Bad: 赵大爷诽谤了她自己。 |
| 先生 → 大娘 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生安慰了他自己。<br>Bad: 王大娘安慰了他自己。 |
| 冯大哥 → 张夫人 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥更喜欢他自己。<br>Bad: 张夫人更喜欢他自己。 |
| 冯大哥 → 徐小姐 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥欺骗了他自己。<br>Bad: 徐小姐欺骗了他自己。 |
| 刘先生 → 周大妈 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生比较喜欢他自己。<br>Bad: 周大妈比较喜欢他自己。 |
| 刘先生 → 王大娘 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生控制了他自己。<br>Bad: 王大娘控制了他自己。 |
| 刘先生 → 陈大姐 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生嫌弃他自己。<br>Bad: 陈大姐嫌弃他自己。 |
| 吴太太 → 刘先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太嫌弃她自己。<br>Bad: 刘先生嫌弃她自己。 |
| 张夫人 → 冯大哥 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人憎恨她自己。<br>Bad: 冯大哥憎恨她自己。 |
| 张夫人 → 刘先生 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人批评了她自己。<br>Bad: 刘先生批评了她自己。 |
| 张夫人 → 李先生 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人喜欢她自己。<br>Bad: 李先生喜欢她自己。 |
| 张夫人 → 赵大爷 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人比较埋怨她自己。<br>Bad: 赵大爷比较埋怨她自己。 |
| 徐小姐 → 王先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐反感她自己。<br>Bad: 王先生反感她自己。 |
| 李先生 → 宋女士 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生支持他自己。<br>Bad: 宋女士支持他自己。 |
| 李先生 → 张夫人 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生欺骗了他自己。<br>Bad: 张夫人欺骗了他自己。 |
| 李太太 → 冯大哥 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太欺骗了她自己。<br>Bad: 冯大哥欺骗了她自己。 |
| 杨大哥 → 宋女士 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥责备了他自己。<br>Bad: 宋女士责备了他自己。 |
| 王先生 → 何太太 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生原谅了他自己。<br>Bad: 何太太原谅了他自己。 |
| 王先生 → 宋女士 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生反感他自己。<br>Bad: 宋女士反感他自己。 |
| 王小姐 → 李先生 | anaphor_gender_agreement | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐称赞了她自己。<br>Bad: 李先生称赞了她自己。 |
| 赵大爷 → 宋女士 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷憎恨他自己。<br>Bad: 宋女士憎恨他自己。 |
| 郑大妈 → 刘先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈最反感她自己。<br>Bad: 刘先生最反感她自己。 |
| 郑大妈 → 张先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈提醒了她自己。<br>Bad: 张先生提醒了她自己。 |
| 陈大姐 → 刘先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐最反感她自己。<br>Bad: 刘先生最反感她自己。 |
| 陈大姐 → 张先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐称赞了她自己。<br>Bad: 张先生称赞了她自己。 |
| 陈大姐 → 李先生 | anaphor_gender_agreement | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐厌恶她自己。<br>Bad: 李先生厌恶她自己。 |

## argument_structure

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| 他 → 糖 | agent_animacy_passive | 4 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那片面包被他吃了。<br>Bad: 那片面包被糖吃了。 |
| 清洗 → 气化 | agent_causative | 4 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 胡大爷清洗了杯子。<br>Bad: 胡大爷气化了杯子。 |
| multiple edits: bad inserts 拿大象的; bad deletes 六头大象 | intransitive_double_obj | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这四位记者寄给了宋女士六头大象。<br>Bad: 这四位记者寄给了拿大象的宋女士。 |
| 跨越 → 融化 | agent_causative | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李太太险些跨越了沙漠。<br>Bad: 李太太险些融化了沙漠。 |
| multiple edits: bad inserts 拿小狗的; bad deletes 许多条小狗 | intransitive_double_obj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那七个吉他手买给了你们许多条小狗。<br>Bad: 那七个吉他手买给了拿小狗的你们。 |
| multiple edits: bad inserts 拿牛的; bad deletes 六头牛 | intransitive_double_obj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位下属寄给了张婶六头牛。<br>Bad: 这位下属寄给了拿牛的张婶。 |
| multiple edits: bad inserts 拿牛的; bad deletes 非常多头牛 | intransitive_double_obj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王五的下属卖给了张三非常多头牛。<br>Bad: 王五的下属卖给了拿牛的张三。 |
| multiple edits: 她几个 -> 拿; bad inserts 的她 | intransitive_double_obj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这四位音乐家卖给了她几个奴隶。<br>Bad: 这四位音乐家卖给了拿奴隶的她。 |
| 哥哥 → 教材 | agent_animacy_subj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 哥哥屠宰了牛。<br>Bad: 教材屠宰了牛。 |
| 唱歌 → 批评 | intransitive_no_obj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位老板唱歌了。<br>Bad: 这位老板批评了。 |
| 我们 → 橙汁 | agent_animacy_passive | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这条裙子被我们清洗了。<br>Bad: 这条裙子被橙汁清洗了。 |
| 检查 → 气化 | agent_causative | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这九个演员检查了腿。<br>Bad: 这九个演员气化了腿。 |
| 清洗 → 凝固 | agent_causative | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 周大妈的老板清洗了杯子。<br>Bad: 周大妈的老板凝固了杯子。 |
| 清蒸 → 融化 | agent_causative | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们清蒸了鸡。<br>Bad: 我们融化了鸡。 |
| 溜走 → 屠宰 | intransitive_no_obj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小王溜走了。<br>Bad: 小王屠宰了。 |
| 演奏员 → 记录片 | agent_animacy_subj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐的演奏员把腿打断了。<br>Bad: 陈大姐的记录片把腿打断了。 |
| 父亲 → 手套 | agent_animacy_subj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 父亲喝了红茶。<br>Bad: 手套喝了红茶。 |
| 看戏 → 预习 | intransitive_no_obj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们看戏了。<br>Bad: 他们预习了。 |
| 睡觉 → 照顾 | intransitive_no_obj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她睡觉了。<br>Bad: 她照顾了。 |
| 警察 → 手套 | agent_animacy_subj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们的警察麻醉了大象。<br>Bad: 他们的手套麻醉了大象。 |
| 跨越 → 气化 | agent_causative | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个领导跨越了海洋。<br>Bad: 这个领导气化了海洋。 |
| 跳舞 → 呵斥 | intransitive_no_obj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小王的领导跳舞了。<br>Bad: 小王的领导呵斥了。 |
| 钢琴家 → 电冰箱 | agent_animacy_subj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 钢琴家把鸡清蒸了。<br>Bad: 电冰箱把鸡清蒸了。 |
| multiple edits: bad inserts 拿儿子的; bad deletes 八个儿子 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那个吉他手递给了杨大哥八个儿子。<br>Bad: 那个吉他手递给了拿儿子的杨大哥。 |
| multiple edits: bad inserts 拿员工的; bad deletes 几位员工 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那个钢琴家送给了吴太太几位员工。<br>Bad: 那个钢琴家送给了拿员工的吴太太。 |
| multiple edits: bad inserts 拿哥哥的; bad deletes 许多个哥哥 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位舞者卖给了他们许多个哥哥。<br>Bad: 那位舞者卖给了拿哥哥的他们。 |
| multiple edits: bad inserts 拿奴隶的; bad deletes 七个奴隶 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外一个朋友卖给了他们七个奴隶。<br>Bad: 另外一个朋友卖给了拿奴隶的他们。 |
| multiple edits: bad inserts 拿妹妹的; bad deletes 十个妹妹 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这六个奴隶借给了赵大爷十个妹妹。<br>Bad: 这六个奴隶借给了拿妹妹的赵大爷。 |
| multiple edits: bad inserts 拿妹妹的; bad deletes 许多个妹妹 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位下属送给了张三许多个妹妹。<br>Bad: 那位下属送给了拿妹妹的张三。 |
| multiple edits: bad inserts 拿姐姐的; bad deletes 几个姐姐 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个朋友寄给了李四几个姐姐。<br>Bad: 这个朋友寄给了拿姐姐的李四。 |
| multiple edits: bad inserts 拿小狗的; bad deletes 七条小狗 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个吉他手送给了王大娘七条小狗。<br>Bad: 这个吉他手送给了拿小狗的王大娘。 |
| multiple edits: bad inserts 拿工人的; bad deletes 五个工人 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你的儿子借给了王五五个工人。<br>Bad: 你的儿子借给了拿工人的王五。 |
| multiple edits: bad inserts 拿消费者的; bad deletes 好几十位消费者 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张婶的下属寄给了吴太太好几十位消费者。<br>Bad: 张婶的下属寄给了拿消费者的吴太太。 |
| multiple edits: bad inserts 拿牛的; bad deletes 八头牛 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外十个领导卖给了她八头牛。<br>Bad: 另外十个领导卖给了拿牛的她。 |
| multiple edits: bad inserts 拿牛的; bad deletes 十头牛 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们的姐妹寄给了我十头牛。<br>Bad: 他们的姐妹寄给了拿牛的我。 |
| multiple edits: bad inserts 拿老板的; bad deletes 几位老板 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥的姐姐借给了你们几位老板。<br>Bad: 冯大哥的姐姐借给了拿老板的你们。 |
| multiple edits: bad inserts 拿老板的; bad deletes 十位老板 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那三个朋友买给了李四十位老板。<br>Bad: 那三个朋友买给了拿老板的李四。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 一只老虎 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外四个记者递给了我们一只老虎。<br>Bad: 另外四个记者递给了拿老虎的我们。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 四只老虎 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外九位学生买给了胡大爷四只老虎。<br>Bad: 另外九位学生买给了拿老虎的胡大爷。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 好几只老虎 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位演奏员卖给了王姨好几只老虎。<br>Bad: 那位演奏员卖给了拿老虎的王姨。 |
| multiple edits: bad inserts 拿蛇的; bad deletes 四条蛇 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位领导送给了她四条蛇。<br>Bad: 这位领导送给了拿蛇的她。 |
| multiple edits: bad inserts 拿记者的; bad deletes 几个记者 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位领导送给了他们几个记者。<br>Bad: 那位领导送给了拿记者的他们。 |
| multiple edits: bad inserts 拿钢琴家的; bad deletes 非常多个钢琴家 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五的母亲递给了张先生非常多个钢琴家。<br>Bad: 王五的母亲递给了拿钢琴家的张先生。 |
| multiple edits: bad inserts 拿音乐家的; bad deletes 九位音乐家 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们的妈妈借给了王小姐九位音乐家。<br>Bad: 我们的妈妈借给了拿音乐家的王小姐。 |
| multiple edits: bad inserts 拿音乐家的; bad deletes 十位音乐家 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王大娘的老师寄给了宋女士十位音乐家。<br>Bad: 王大娘的老师寄给了拿音乐家的宋女士。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 一只鸭 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小明的哥哥寄给了何太太一只鸭。<br>Bad: 小明的哥哥寄给了拿鸭的何太太。 |
| multiple edits: 他七个 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们的姐妹递给了他七个小孩。<br>Bad: 他们的姐妹递给了拿小孩的他。 |
| multiple edits: 你九位 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这六个打工人买给了你九位打工人。<br>Bad: 这六个打工人买给了拿打工人的你。 |
| multiple edits: 你许多头 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 郑大妈的哥哥借给了你许多头大象。<br>Bad: 郑大妈的哥哥借给了拿大象的你。 |
| multiple edits: 她十位 -> 拿; bad inserts 的她 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这四位吉他手卖给了她十位消费者。<br>Bad: 这四位吉他手卖给了拿消费者的她。 |
| multiple edits: 我们非常多位 -> 拿; bad inserts 的我们 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外四个顾客寄给了我们非常多位打工人。<br>Bad: 另外四个顾客寄给了拿打工人的我们。 |
| multiple edits: 我十位 -> 拿; bad inserts 的我 | intransitive_double_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位领导送给了我十位上级。<br>Bad: 这位领导送给了拿上级的我。 |
| 上级 → 花卷 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们的上级吃了橘子。<br>Bad: 他们的花卷吃了橘子。 |
| 上级 → 香蕉 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她的上级把花卷吃了。<br>Bad: 她的香蕉把花卷吃了。 |
| 下属 → 啤酒 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 下属把鱼清蒸了。<br>Bad: 啤酒把鱼清蒸了。 |
| 下属 → 手套 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生的下属清蒸了鸭。<br>Bad: 李先生的手套清蒸了鸭。 |
| 下属 → 橙汁 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 下属领养了小猫。<br>Bad: 橙汁领养了小猫。 |
| 他 → 书 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外一块糖果被他吃了。<br>Bad: 另外一块糖果被书吃了。 |
| 他们 → 被子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这桶方便面被他们买了。<br>Bad: 这桶方便面被被子买了。 |
| 何太太 → 开瓶器 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个耳朵被何太太打断了。<br>Bad: 这个耳朵被开瓶器打断了。 |
| 何太太 → 玻璃珠 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那串香蕉被何太太吃了。<br>Bad: 那串香蕉被玻璃珠吃了。 |
| 何太太 → 电视机 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那瓶矿泉水被何太太喝了。<br>Bad: 那瓶矿泉水被电视机喝了。 |
| 何太太 → 饮料瓶 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那六部电影被何太太拍摄了。<br>Bad: 那六部电影被饮料瓶拍摄了。 |
| 你们 → 坚果 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那五头牛被你们屠宰了。<br>Bad: 那五头牛被坚果屠宰了。 |
| 你们 → 山洞 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外十头牛被你们屠宰了。<br>Bad: 另外十头牛被山洞屠宰了。 |
| 你们 → 电影 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这头大象被你们麻醉了。<br>Bad: 这头大象被电影麻醉了。 |
| 你们 → 糖果 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这张桌子被你们吃了。<br>Bad: 这张桌子被糖果吃了。 |
| 健身 → 厌恶 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李先生健身了。<br>Bad: 李先生厌恶了。 |
| 健身 → 治疗 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们的哥哥健身了。<br>Bad: 我们的哥哥治疗了。 |
| 健身 → 演奏 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 胡大爷健身了。<br>Bad: 胡大爷演奏了。 |
| 偷听 → 嘉奖 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他偷听了。<br>Bad: 他嘉奖了。 |
| 偷听 → 欺骗 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们偷听了。<br>Bad: 你们欺骗了。 |
| 偷听 → 诽谤 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王大娘的领导偷听了。<br>Bad: 王大娘的领导诽谤了。 |
| 儿 → 裙 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们的儿子烧过鸭了。<br>Bad: 他们的裙子烧过鸭了。 |
| 兄弟 → 双簧 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 兄弟把被子盖了。<br>Bad: 双簧把被子盖了。 |
| 入睡 → 嘉奖 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们入睡了。<br>Bad: 我们嘉奖了。 |
| 入睡 → 宠爱 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我的朋友入睡了。<br>Bad: 我的朋友宠爱了。 |
| 入睡 → 批判 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这五位钢琴家入睡了。<br>Bad: 这五位钢琴家批判了。 |
| 入睡 → 相信 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你入睡了。<br>Bad: 你相信了。 |
| 冯大哥 → 饮料瓶 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那瓶啤酒被冯大哥喝了。<br>Bad: 那瓶啤酒被饮料瓶喝了。 |
| 出发 → 喜欢 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个老板出发了。<br>Bad: 这个老板喜欢了。 |
| 出发 → 抨击 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐出发了。<br>Bad: 陈大姐抨击了。 |
| 出发 → 推崇 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷的领导出发了。<br>Bad: 赵大爷的领导推崇了。 |
| 出发 → 检查 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这九个朋友出发了。<br>Bad: 这九个朋友检查了。 |
| 出发 → 照顾 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位顾客出发了。<br>Bad: 这位顾客照顾了。 |
| 出发 → 表扬 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我的兄弟出发了。<br>Bad: 我的兄弟表扬了。 |
| 出发 → 观看 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太的老板出发了。<br>Bad: 何太太的老板观看了。 |
| 刘先生 → 矿泉水 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这片面包被刘先生买了。<br>Bad: 这片面包被矿泉水买了。 |
| 创作 → 存在 | agent_causative | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个儿子创作了小说。<br>Bad: 这个儿子存在了小说。 |
| 制作 → 气化 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李四几乎制作了电影。<br>Bad: 李四几乎气化了电影。 |
| 制作 → 蒸发 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张先生的领导制作了手账。<br>Bad: 张先生的领导蒸发了手账。 |
| 去 → 盖 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外两个打工人去了。<br>Bad: 另外两个打工人盖了。 |
| 司机 → 戏曲 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 司机把小提琴拉了。<br>Bad: 戏曲把小提琴拉了。 |
| 叹息 → 欺骗 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那两位老师叹息了。<br>Bad: 那两位老师欺骗了。 |
| 吉他手 → 热水器 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 吉他手预习了教材。<br>Bad: 热水器预习了教材。 |
| 吉他手 → 电视机 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她的吉他手炖过鱼了。<br>Bad: 她的电视机炖过鱼了。 |
| 同事 → 手套 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 同事把巧克力吃了。<br>Bad: 手套把巧克力吃了。 |
| 同事 → 椅子 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我的同事观看过电影了。<br>Bad: 我的椅子观看过电影了。 |
| 同事 → 白酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你的同事喝过啤酒了。<br>Bad: 你的白酒喝过啤酒了。 |
| 听课 → 拥护 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们的弟弟听课了。<br>Bad: 她们的弟弟拥护了。 |
| 启程 → 拥护 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷启程了。<br>Bad: 赵大爷拥护了。 |
| 吴太太 → 开瓶器 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那条被子被吴太太盖了。<br>Bad: 那条被子被开瓶器盖了。 |
| 周大妈 → 电冰箱 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那杯牛奶被周大妈喝了。<br>Bad: 那杯牛奶被电冰箱喝了。 |
| 呼吸 → 捕捉 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她的姐姐呼吸了。<br>Bad: 她的姐姐捕捉了。 |
| 品茶 → 支持 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那个妹妹品茶了。<br>Bad: 那个妹妹支持了。 |
| 品茶 → 称赞 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这九个罪犯品茶了。<br>Bad: 这九个罪犯称赞了。 |
| 哥哥 → 橘子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 哥哥打断过腿了。<br>Bad: 橘子打断过腿了。 |
| 哭 → 学 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥哭了。<br>Bad: 杨大哥学了。 |
| 哭 → 烧 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外十个姐姐哭了。<br>Bad: 另外十个姐姐烧了。 |
| 哭 → 煮 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王姨哭了。<br>Bad: 王姨煮了。 |
| 唱歌 → 伤害 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五唱歌了。<br>Bad: 王五伤害了。 |
| 唱歌 → 爆炒 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们唱歌了。<br>Bad: 你们爆炒了。 |
| 唱歌 → 爱戴 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们的儿子唱歌了。<br>Bad: 你们的儿子爱戴了。 |
| 坐下 → 取代 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李四坐下了。<br>Bad: 李四取代了。 |
| 坐下 → 埋怨 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位司机坐下了。<br>Bad: 这位司机埋怨了。 |
| 坐下 → 建立 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外六个弟弟坐下了。<br>Bad: 另外六个弟弟建立了。 |
| 坐下 → 拥护 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位钢琴家坐下了。<br>Bad: 这位钢琴家拥护了。 |
| 坐下 → 捕捉 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她坐下了。<br>Bad: 她捕捉了。 |
| 坐下 → 推崇 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们的妈妈坐下了。<br>Bad: 她们的妈妈推崇了。 |
| 坐下 → 约束 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王大娘的母亲坐下了。<br>Bad: 王大娘的母亲约束了。 |
| 奴隶 → 白酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷的奴隶把录像带看了。<br>Bad: 胡大爷的白酒把录像带看了。 |
| 奴隶 → 裙子 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 奴隶创作过小说了。<br>Bad: 裙子创作过小说了。 |
| 她 → 书 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那四头牛被她屠宰了。<br>Bad: 那四头牛被书屠宰了。 |
| 她 → 糖 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那串香蕉被她买了。<br>Bad: 那串香蕉被糖买了。 |
| 她们 → 手套 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外六条蛇被她们炖了。<br>Bad: 另外六条蛇被手套炖了。 |
| 她们 → 椅子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这三块巧克力被她们吃了。<br>Bad: 这三块巧克力被椅子吃了。 |
| 她们 → 橙汁 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这条鱼被她们捕捉了。<br>Bad: 这条鱼被橙汁捕捉了。 |
| 她们 → 蛋糕 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那条裤子被她们看了。<br>Bad: 那条裤子被蛋糕看了。 |
| 她们 → 袜子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这四本教材被她们清洗了。<br>Bad: 这四本教材被袜子清洗了。 |
| 她们 → 被子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这张桌子被她们搬了。<br>Bad: 这张桌子被被子搬了。 |
| 她们 → 裤子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这部手账被她们写了。<br>Bad: 这部手账被裤子写了。 |
| 她们 → 面包 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外八把椅子被她们搬了。<br>Bad: 另外八把椅子被面包搬了。 |
| 妈妈 → 手套 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们的妈妈把飞机驾驶了。<br>Bad: 你们的手套把飞机驾驶了。 |
| 妹妹 → 手套 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 妹妹爆炒过鸭了。<br>Bad: 手套爆炒过鸭了。 |
| 妹妹 → 火山 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 妹妹弹了玻璃珠。<br>Bad: 火山弹了玻璃珠。 |
| 妹妹 → 红酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 妹妹拍摄了电影。<br>Bad: 红酒拍摄了电影。 |
| 姐妹 → 啤酒 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 姐妹把鸡爆炒了。<br>Bad: 啤酒把鸡爆炒了。 |
| 姐妹 → 白酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 姐妹制作过电影了。<br>Bad: 白酒制作过电影了。 |
| 姐姐 → 啤酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 姐姐把电影拍摄了。<br>Bad: 啤酒把电影拍摄了。 |
| 姐姐 → 手套 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们的姐姐拍摄过电视剧了。<br>Bad: 你们的手套拍摄过电视剧了。 |
| 姐姐 → 红茶 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 姐姐喝了白酒。<br>Bad: 红茶喝了白酒。 |
| 姐姐 → 衣服 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 姐姐麻醉过老虎了。<br>Bad: 衣服麻醉过老虎了。 |
| 学生 → 葡萄 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们的学生吹了双簧。<br>Bad: 你们的葡萄吹了双簧。 |
| 宋女士 → 俱乐部 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那八条小狗被宋女士领养了。<br>Bad: 那八条小狗被俱乐部领养了。 |
| 宋女士 → 录像带 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这桶啤酒被宋女士喝了。<br>Bad: 这桶啤酒被录像带喝了。 |
| 小孩 → 手套 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小王的小孩吹过双簧了。<br>Bad: 小王的手套吹过双簧了。 |
| 小明 → 可乐 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那一条腿被小明检查了。<br>Bad: 那一条腿被可乐检查了。 |
| 小明 → 手套 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外四块糖被小明买了。<br>Bad: 另外四块糖被手套买了。 |
| 小明 → 椅子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外四只脚被小明打断了。<br>Bad: 另外四只脚被椅子打断了。 |
| 小明 → 电影 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外八桶矿泉水被小明喝了。<br>Bad: 另外八桶矿泉水被电影喝了。 |
| 小明 → 苹果 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外九头牛被小明捕捉了。<br>Bad: 另外九头牛被苹果捕捉了。 |
| 小王 → 椅子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外八片面包被小王吃了。<br>Bad: 另外八片面包被椅子吃了。 |
| 工人 → 作业 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们的工人把教材预习了。<br>Bad: 他们的作业把教材预习了。 |
| 工人 → 可乐 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐的工人预习过教材了。<br>Bad: 陈大姐的可乐预习过教材了。 |
| 工人 → 咖啡 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 工人预习过教材了。<br>Bad: 咖啡预习过教材了。 |
| 工人 → 啤酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们的工人吃了蛋糕。<br>Bad: 我们的啤酒吃了蛋糕。 |
| 工人 → 山洞 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 工人把鸭清蒸了。<br>Bad: 山洞把鸭清蒸了。 |
| 工人 → 手账 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们的工人检查了耳朵。<br>Bad: 他们的手账检查了耳朵。 |
| 工人 → 面包 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈的工人开了火车。<br>Bad: 周大妈的面包开了火车。 |
| 弟弟 → 可乐 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 宋女士的弟弟把鱼清蒸了。<br>Bad: 宋女士的可乐把鱼清蒸了。 |
| 张三 → 手套 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那本教材被张三预习了。<br>Bad: 那本教材被手套预习了。 |
| 张先生 → 动作片 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外六个领导被张先生嘉奖了。<br>Bad: 另外六个领导被动作片嘉奖了。 |
| 张先生 → 电冰箱 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那部教材被张先生写了。<br>Bad: 那部教材被电冰箱写了。 |
| 张夫人 → 照相馆 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那头牛被张夫人爆炒了。<br>Bad: 那头牛被照相馆爆炒了。 |
| 张婶 → 小说 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这五瓶橙汁被张婶买了。<br>Bad: 这五瓶橙汁被小说买了。 |
| 张婶 → 手套 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这八只鸡被张婶屠宰了。<br>Bad: 这八只鸡被手套屠宰了。 |
| 张婶 → 红酒 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那八本小说被张婶观看了。<br>Bad: 那八本小说被红酒观看了。 |
| 张婶 → 美声 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那杯牛奶被张婶买了。<br>Bad: 那杯牛奶被美声买了。 |
| 张婶 → 苹果 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这本教材被张婶喝了。<br>Bad: 这本教材被苹果喝了。 |
| 张婶 → 裙子 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那条鱼被张婶爆炒了。<br>Bad: 那条鱼被裙子爆炒了。 |
| 徐小姐 → 收音机 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位下属被徐小姐控制了。<br>Bad: 这位下属被收音机控制了。 |
| 徐小姐 → 玻璃珠 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个头被徐小姐包扎了。<br>Bad: 这个头被玻璃珠包扎了。 |
| 微笑 → 喜欢 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他的学生微笑了。<br>Bad: 他的学生喜欢了。 |
| 微笑 → 埋怨 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他微笑了。<br>Bad: 他埋怨了。 |
| 微笑 → 批评 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们的学生微笑了。<br>Bad: 你们的学生批评了。 |
| 微笑 → 爱护 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她微笑了。<br>Bad: 她爱护了。 |
| 微笑 → 称赞 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那个打工人微笑了。<br>Bad: 那个打工人称赞了。 |
| 我们 → 手账 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那七张桌子被我们搬了。<br>Bad: 那七张桌子被手账搬了。 |
| 我们 → 火山 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那四条鱼被我们烧了。<br>Bad: 那四条鱼被火山烧了。 |
| 我们 → 蛋糕 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那一只脚被我们包扎了。<br>Bad: 那一只脚被蛋糕包扎了。 |
| 打工人 → 奏鸣曲 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们的打工人把录像带看了。<br>Bad: 我们的奏鸣曲把录像带看了。 |
| 打工人 → 方便面 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李四的打工人清蒸了鸡。<br>Bad: 李四的方便面清蒸了鸡。 |
| 打架 → 相信 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们打架了。<br>Bad: 她们相信了。 |
| 拍摄 → 存在 | agent_causative | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥的女儿拍摄了电影。<br>Bad: 杨大哥的女儿存在了电影。 |
| 拍摄 → 蒸发 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 杨大哥的兄弟拍摄了电影。<br>Bad: 杨大哥的兄弟蒸发了电影。 |
| 捕捉 → 气化 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五险些捕捉了鸭。<br>Bad: 王五险些气化了鸭。 |
| 有点 → 无心 | agent_animacy_adv | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 几桶啤酒有点过期了。<br>Bad: 几桶啤酒无心过期了。 |
| 朋友 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我的朋友唱了小调。<br>Bad: 我的杯子唱了小调。 |
| 朋友 → 白酒 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 朋友屠宰过牛了。<br>Bad: 白酒屠宰过牛了。 |
| 朋友 → 蛋糕 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 朋友预习了教材。<br>Bad: 蛋糕预习了教材。 |
| 朋友 → 衣服 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 朋友麻醉过老虎了。<br>Bad: 衣服麻醉过老虎了。 |
| 李先生 → 充电器 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这六桶方便面被李先生吃了。<br>Bad: 这六桶方便面被充电器吃了。 |
| 李先生 → 冰红茶 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外十个充电器被李先生盖了。<br>Bad: 另外十个充电器被冰红茶盖了。 |
| 李先生 → 巧克力 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那只袜子被李先生吃了。<br>Bad: 那只袜子被巧克力吃了。 |
| 李先生 → 派出所 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那部小说被李先生观看了。<br>Bad: 那部小说被派出所观看了。 |
| 李先生 → 蛋炒饭 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个头被李先生包扎了。<br>Bad: 那个头被蛋炒饭包扎了。 |
| 李四 → 牛奶 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这瓶红酒被李四买了。<br>Bad: 这瓶红酒被牛奶买了。 |
| 李太太 → 热水器 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那片面包被李太太买了。<br>Bad: 那片面包被热水器买了。 |
| 检查 → 蒸发 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张先生检查了鼻子。<br>Bad: 张先生蒸发了鼻子。 |
| 清洗 → 蒸发 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五的母亲几乎清洗了杯子。<br>Bad: 王五的母亲几乎蒸发了杯子。 |
| 清蒸 → 存在 | agent_causative | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位空姐几乎清蒸了鱼。<br>Bad: 这位空姐几乎存在了鱼。 |
| 清蒸 → 消失 | agent_causative | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈清蒸了鸭。<br>Bad: 周大妈消失了鸭。 |
| 游泳 → 取缔 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他游泳了。<br>Bad: 他取缔了。 |
| 游泳 → 批评 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她的朋友游泳了。<br>Bad: 她的朋友批评了。 |
| 游泳 → 称赞 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个演员游泳了。<br>Bad: 这个演员称赞了。 |
| 游泳 → 鼓励 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那位母亲游泳了。<br>Bad: 那位母亲鼓励了。 |
| 演奏员 → 充电器 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 演奏员麻醉过大象了。<br>Bad: 充电器麻醉过大象了。 |
| 演奏员 → 开瓶器 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他的演奏员观看了电影。<br>Bad: 他的开瓶器观看了电影。 |
| 演奏员 → 电视机 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 演奏员创作过小说了。<br>Bad: 电视机创作过小说了。 |
| 爆炒 → 蒸发 | agent_causative | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们差点儿爆炒了鱼。<br>Bad: 我们差点儿蒸发了鱼。 |
| 爬行 → 相信 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那三个舞者爬行了。<br>Bad: 那三个舞者相信了。 |
| 父亲 → 袜子 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张夫人的父亲炖过鸭了。<br>Bad: 张夫人的袜子炖过鸭了。 |
| 父亲 → 裙子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 父亲看了漫画。<br>Bad: 裙子看了漫画。 |
| 爸爸 → 咖啡 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 爸爸炖了鱼。<br>Bad: 咖啡炖了鱼。 |
| 爸爸 → 教材 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 刘先生的爸爸盖了被子。<br>Bad: 刘先生的教材盖了被子。 |
| 爸爸 → 椅子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 爸爸驾驶过货车了。<br>Bad: 椅子驾驶过货车了。 |
| 爸爸 → 糖果 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 爸爸创作了漫画。<br>Bad: 糖果创作了漫画。 |
| 王五 → 教材 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那只手套被王五预习了。<br>Bad: 那只手套被教材预习了。 |
| 王五 → 椅子 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那串香蕉被王五买了。<br>Bad: 那串香蕉被椅子买了。 |
| 王先生 → 热水器 | agent_animacy_passive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这本手账被王先生制作了。<br>Bad: 这本手账被热水器制作了。 |
| 王先生 → 电冰箱 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这片面包被王先生买了。<br>Bad: 这片面包被电冰箱买了。 |
| 王大娘 → 电冰箱 | agent_animacy_passive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那条裙子被王大娘盖了。<br>Bad: 那条裙子被电冰箱盖了。 |
| 看戏 → 包扎 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位演员看戏了。<br>Bad: 这位演员包扎了。 |
| 睡觉 → 取代 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这八位演员睡觉了。<br>Bad: 这八位演员取代了。 |
| 睡觉 → 回到 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那两位演员睡觉了。<br>Bad: 那两位演员回到了。 |
| 睡觉 → 憎恨 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那九位上级睡觉了。<br>Bad: 那九位上级憎恨了。 |
| 睡觉 → 批判 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个记者睡觉了。<br>Bad: 这个记者批判了。 |
| 睡觉 → 登上 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外一位打工人睡觉了。<br>Bad: 另外一位打工人登上了。 |
| 睡觉 → 观看 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们睡觉了。<br>Bad: 她们观看了。 |
| 空姐 → 手套 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张婶的空姐预习了教材。<br>Bad: 张婶的手套预习了教材。 |
| 空姐 → 电影 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 空姐跨越过沙漠了。<br>Bad: 电影跨越过沙漠了。 |
| 空姐 → 裤子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 空姐把手打断了。<br>Bad: 裤子把手打断了。 |
| 站立 → 喜欢 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷站立了。<br>Bad: 赵大爷喜欢了。 |
| 站立 → 嘉奖 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张先生的上级站立了。<br>Bad: 张先生的上级嘉奖了。 |
| 站立 → 夸奖 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐站立了。<br>Bad: 陈大姐夸奖了。 |
| 站立 → 捕捉 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 徐小姐站立了。<br>Bad: 徐小姐捕捉了。 |
| 站立 → 支持 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们站立了。<br>Bad: 他们支持了。 |
| 站立 → 表扬 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生站立了。<br>Bad: 王先生表扬了。 |
| 站立 → 重建 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥站立了。<br>Bad: 杨大哥重建了。 |
| 罪犯 → 教材 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张三的罪犯弹了玻璃珠。<br>Bad: 张三的教材弹了玻璃珠。 |
| 罪犯 → 裙子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐的罪犯爆炒过鸡了。<br>Bad: 徐小姐的裙子爆炒过鸡了。 |
| 老师 → 可乐 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 老师把被子盖了。<br>Bad: 可乐把被子盖了。 |
| 老师 → 啤酒 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 老师弹了玻璃珠。<br>Bad: 啤酒弹了玻璃珠。 |
| 老师 → 橙汁 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小明的老师演奏了华尔兹。<br>Bad: 小明的橙汁演奏了华尔兹。 |
| 老板 → 啤酒 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 老板包扎了手。<br>Bad: 啤酒包扎了手。 |
| 老板 → 裙子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 老板爆炒过鱼了。<br>Bad: 裙子爆炒过鱼了。 |
| 老板 → 视频 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们的老板唱过歌了。<br>Bad: 他们的视频唱过歌了。 |
| 舞者 → 作业 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他的舞者把咖啡喝了。<br>Bad: 他的作业把咖啡喝了。 |
| 舞者 → 蛋糕 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你的舞者看了小说。<br>Bad: 你的蛋糕看了小说。 |
| 舞者 → 裤子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 舞者把笛子吹了。<br>Bad: 裤子把笛子吹了。 |
| 记者 → 桌子 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 记者制作过电影了。<br>Bad: 桌子制作过电影了。 |
| 记者 → 白酒 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们的记者拍摄过电影了。<br>Bad: 她们的白酒拍摄过电影了。 |
| 记者 → 视频 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你的记者观看过电影了。<br>Bad: 你的视频观看过电影了。 |
| 走路 → 偷听 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王大娘的儿子走路了。<br>Bad: 王大娘的儿子偷听了。 |
| 走路 → 宠爱 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她的老板走路了。<br>Bad: 她的老板宠爱了。 |
| 走路 → 推崇 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位工人走路了。<br>Bad: 那位工人推崇了。 |
| 起飞 → 包扎 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王大娘的姐妹起飞了。<br>Bad: 王大娘的姐妹包扎了。 |
| 起飞 → 取缔 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五的姐姐起飞了。<br>Bad: 王五的姐姐取缔了。 |
| 起飞 → 批判 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们起飞了。<br>Bad: 你们批判了。 |
| 起飞 → 麻醉 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥起飞了。<br>Bad: 冯大哥麻醉了。 |
| 跑步 → 呵斥 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐的下属跑步了。<br>Bad: 徐小姐的下属呵斥了。 |
| 跑步 → 欺骗 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐跑步了。<br>Bad: 陈大姐欺骗了。 |
| 跳舞 → 排挤 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你的下属跳舞了。<br>Bad: 你的下属排挤了。 |
| 跳舞 → 爱戴 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们跳舞了。<br>Bad: 他们爱戴了。 |
| 跳舞 → 知道 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张三的父亲跳舞了。<br>Bad: 张三的父亲知道了。 |
| 跳舞 → 称赞 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们跳舞了。<br>Bad: 我们称赞了。 |
| 躺下 → 反感 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 周大妈躺下了。<br>Bad: 周大妈反感了。 |
| 躺下 → 推崇 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 郑大妈躺下了。<br>Bad: 郑大妈推崇了。 |
| 过来 → 批评 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们过来了。<br>Bad: 我们批评了。 |
| 过来 → 鼓励 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张先生过来了。<br>Bad: 张先生鼓励了。 |
| 运动 → 约束 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那个记者运动了。<br>Bad: 那个记者约束了。 |
| 钢琴家 → 协奏曲 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们的钢琴家唱了歌。<br>Bad: 我们的协奏曲唱了歌。 |
| 钢琴家 → 饮料瓶 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 钢琴家吃过葡萄了。<br>Bad: 饮料瓶吃过葡萄了。 |
| 音乐家 → 电视机 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 音乐家创作了小说。<br>Bad: 电视机创作了小说。 |
| 顾客 → 电影 | agent_animacy_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 顾客把奏鸣曲演奏了。<br>Bad: 电影把奏鸣曲演奏了。 |
| 顾客 → 葡萄 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 顾客屠宰过牛了。<br>Bad: 葡萄屠宰过牛了。 |
| 领导 → 糖果 | agent_animacy_subj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 领导把海洋跨越了。<br>Bad: 糖果把海洋跨越了。 |
| 颤抖 → 埋怨 | intransitive_no_obj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五颤抖了。<br>Bad: 王五埋怨了。 |
| 颤抖 → 驾驶 | intransitive_no_obj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那十个司机颤抖了。<br>Bad: 那十个司机驾驶了。 |
| 驾驶 → 融化 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张夫人驾驶了飞机。<br>Bad: 张夫人融化了飞机。 |
| 麻醉 → 凝固 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们麻醉了大象。<br>Bad: 她们凝固了大象。 |
| 麻醉 → 气化 | agent_causative | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐麻醉了老虎。<br>Bad: 徐小姐气化了老虎。 |
| 你们 → 手套 | agent_animacy_passive | 7 | 1.0000 | 0.1429 | +0.8571 | 0.0000 | Good: 另外九把椅子被你们吃了。<br>Bad: 另外九把椅子被手套吃了。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 好几只鸡 | intransitive_double_obj | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 那位钢琴家卖给了我们好几只鸡。<br>Bad: 那位钢琴家卖给了拿鸡的我们。 |
| 我们 → 手套 | agent_animacy_passive | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 那十片面包被我们吃了。<br>Bad: 那十片面包被手套吃了。 |
| 拍摄 → 变化 | agent_causative | 4 | 0.7500 | 0.0000 | +0.7500 | 0.0000 | Good: 另外三个女儿几乎拍摄了电影。<br>Bad: 另外三个女儿几乎变化了电影。 |
| bad deletes 她 | agent_deletion | 33 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 冯大哥，那个秘密跟她没有关系。<br>Bad: 冯大哥，那个秘密跟没有关系。 |
| multiple edits: bad inserts 拿大象的; bad deletes 五头大象 | intransitive_double_obj | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 这个弟弟送给了宋女士五头大象。<br>Bad: 这个弟弟送给了拿大象的宋女士。 |
| multiple edits: bad inserts 拿大象的; bad deletes 许多头大象 | intransitive_double_obj | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 那个儿子卖给了她们许多头大象。<br>Bad: 那个儿子卖给了拿大象的她们。 |
| multiple edits: bad inserts 拿牛的; bad deletes 两头牛 | intransitive_double_obj | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 这两个小孩买给了王大娘两头牛。<br>Bad: 这两个小孩买给了拿牛的王大娘。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 几只鸡 | intransitive_double_obj | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 王姨的下属借给了陈大姐几只鸡。<br>Bad: 王姨的下属借给了拿鸡的陈大姐。 |
| 他们 → 杯子 | agent_animacy_passive | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 那四本教材被他们清洗了。<br>Bad: 那四本教材被杯子清洗了。 |
| 小王 → 手套 | agent_animacy_passive | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 这六张桌子被小王预习了。<br>Bad: 这六张桌子被手套预习了。 |
| 王大娘 → 电视机 | agent_animacy_passive | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 这把椅子被王大娘搬了。<br>Bad: 这把椅子被电视机搬了。 |
| 预习 → 融化 | agent_causative | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 她们几乎预习了教材。<br>Bad: 她们几乎融化了教材。 |
| 创作 → 变化 | agent_causative | 3 | 0.6667 | 0.0000 | +0.6667 | 0.0000 | Good: 这个服务员险些创作了漫画。<br>Bad: 这个服务员险些变化了漫画。 |
| 她们 → 杯子 | agent_animacy_passive | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 那张桌子被她们盖了。<br>Bad: 那张桌子被杯子盖了。 |
| 我们 → 糖果 | agent_animacy_passive | 3 | 0.6667 | 0.0000 | +0.6667 | 0.0000 | Good: 那十头大象被我们吃了。<br>Bad: 那十头大象被糖果吃了。 |
| 捕捉 → 融化 | agent_causative | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 你捕捉了鱼。<br>Bad: 你融化了鱼。 |
| 检查 → 融化 | agent_causative | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 李先生的领导差点儿检查了鼻子。<br>Bad: 李先生的领导差点儿融化了鼻子。 |
| 拍摄 → 气化 | agent_causative | 4 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 李四的上级几乎拍摄了电影。<br>Bad: 李四的上级几乎气化了电影。 |
| 跨越 → 蒸发 | agent_causative | 4 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 冯大哥跨越了沙漠。<br>Bad: 冯大哥蒸发了沙漠。 |
| multiple edits: bad inserts 拿司机的; bad deletes 九位司机 | intransitive_double_obj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 另外六位同事卖给了何太太九位司机。<br>Bad: 另外六位同事卖给了拿司机的何太太。 |
| multiple edits: bad inserts 拿大象的; bad deletes 十几头大象 | intransitive_double_obj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 这个奴隶寄给了徐小姐十几头大象。<br>Bad: 这个奴隶寄给了拿大象的徐小姐。 |
| multiple edits: bad inserts 拿老板的; bad deletes 三位老板 | intransitive_double_obj | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 这三位老师送给了我们三位老板。<br>Bad: 这三位老师送给了拿老板的我们。 |
| multiple edits: bad inserts 拿老板的; bad deletes 两位老板 | intransitive_double_obj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 另外七个服务员寄给了你们两位老板。<br>Bad: 另外七个服务员寄给了拿老板的你们。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 七只老虎 | intransitive_double_obj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这一位音乐家借给了王姨七只老虎。<br>Bad: 这一位音乐家借给了拿老虎的王姨。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 一只鸡 | intransitive_double_obj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 这位学生寄给了吴太太一只鸡。<br>Bad: 这位学生寄给了拿鸡的吴太太。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 十几只鸡 | intransitive_double_obj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那个顾客卖给了郑大妈十几只鸡。<br>Bad: 那个顾客卖给了拿鸡的郑大妈。 |
| multiple edits: 她八头 -> 拿; bad inserts 的她 | intransitive_double_obj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 另外五位上级卖给了她八头大象。<br>Bad: 另外五位上级卖给了拿大象的她。 |
| 他们 → 教材 | agent_animacy_passive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这本手账被他们制作了。<br>Bad: 这本手账被教材制作了。 |
| 他们 → 袜子 | agent_animacy_passive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 另外七本书被他们写了。<br>Bad: 另外七本书被袜子写了。 |
| 何太太 → 热水器 | agent_animacy_passive | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 这个想法被何太太辩护了。<br>Bad: 这个想法被热水器辩护了。 |
| 你们 → 袜子 | agent_animacy_passive | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 那一本手账被你们制作了。<br>Bad: 那一本手账被袜子制作了。 |
| 你们 → 裙子 | agent_animacy_passive | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 这片面包被你们买了。<br>Bad: 这片面包被裙子买了。 |
| 兄弟 → 衣服 | agent_animacy_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 兄弟把蛇捕捉了。<br>Bad: 衣服把蛇捕捉了。 |
| 入睡 → 批评 | intransitive_no_obj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 张三入睡了。<br>Bad: 张三批评了。 |
| 入睡 → 维护 | intransitive_no_obj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 李四入睡了。<br>Bad: 李四维护了。 |
| 冯大哥 → 充电器 | agent_animacy_passive | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 这把椅子被冯大哥搬了。<br>Bad: 这把椅子被充电器搬了。 |
| 可能 → 无心 | agent_animacy_adv | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 好几个开瓶器可能故障了。<br>Bad: 好几个开瓶器无心故障了。 |
| 吉他手 → 充电器 | agent_animacy_subj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 她们的吉他手把杯子清洗了。<br>Bad: 她们的充电器把杯子清洗了。 |
| 呼吸 → 相信 | intransitive_no_obj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 王大娘呼吸了。<br>Bad: 王大娘相信了。 |
| 妈妈 → 袜子 | agent_animacy_subj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 妈妈观看了电影。<br>Bad: 袜子观看了电影。 |
| 妹妹 → 漫画 | agent_animacy_subj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 妹妹把牛屠宰了。<br>Bad: 漫画把牛屠宰了。 |
| 小明 → 杯子 | agent_animacy_passive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 那九桶方便面被小明买了。<br>Bad: 那九桶方便面被杯子买了。 |
| 小明 → 红酒 | agent_animacy_passive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 那八只鸡被小明吃了。<br>Bad: 那八只鸡被红酒吃了。 |
| 小明 → 袜子 | agent_animacy_passive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 这八本漫画被小明观看了。<br>Bad: 这八本漫画被袜子观看了。 |
| 小王 → 杯子 | agent_animacy_passive | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 这张桌子被小王清洗了。<br>Bad: 这张桌子被杯子清洗了。 |
| 弟弟 → 教材 | agent_animacy_subj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 宋女士的弟弟喝了白酒。<br>Bad: 宋女士的教材喝了白酒。 |
| 张婶 → 袜子 | agent_animacy_passive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 那部小说被张婶观看了。<br>Bad: 那部小说被袜子观看了。 |
| 我们 → 红酒 | agent_animacy_passive | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 这瓶橙汁被我们喝了。<br>Bad: 这瓶橙汁被红酒喝了。 |
| 我们 → 袜子 | agent_animacy_passive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 另外四瓶可乐被我们喝了。<br>Bad: 另外四瓶可乐被袜子喝了。 |
| 打工人 → 饮料瓶 | agent_animacy_subj | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 她们的打工人领养过小狗了。<br>Bad: 她们的饮料瓶领养过小狗了。 |
| 拍摄 → 融化 | agent_causative | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 她拍摄了动画片。<br>Bad: 她融化了动画片。 |
| 李先生 → 动画片 | agent_animacy_passive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那七片面包被李先生吃了。<br>Bad: 那七片面包被动画片吃了。 |
| 杨大哥 → 开瓶器 | agent_animacy_passive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这七条腿被杨大哥打断了。<br>Bad: 这七条腿被开瓶器打断了。 |
| 检查 → 凝固 | agent_causative | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 李四的朋友差点儿检查了肚子。<br>Bad: 李四的朋友差点儿凝固了肚子。 |
| 检查 → 存在 | agent_causative | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你们检查了胃。<br>Bad: 你们存在了胃。 |
| 演员 → 电影 | agent_animacy_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 周大妈的演员打断了鼻子。<br>Bad: 周大妈的电影打断了鼻子。 |
| 演员 → 面包 | agent_animacy_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 她们的演员喝了牛奶。<br>Bad: 她们的面包喝了牛奶。 |
| 演奏 → 融化 | agent_causative | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 那六个顾客险些演奏了歌曲。<br>Bad: 那六个顾客险些融化了歌曲。 |
| 爆炒 → 凝固 | agent_causative | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 他们险些爆炒了鱼。<br>Bad: 他们险些凝固了鱼。 |
| 爆炒 → 融化 | agent_causative | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 我差点儿爆炒了鸡。<br>Bad: 我差点儿融化了鸡。 |
| 王姨 → 手套 | agent_animacy_passive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这三把椅子被王姨搬了。<br>Bad: 这三把椅子被手套搬了。 |
| 看戏 → 批评 | intransitive_no_obj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 这位记者看戏了。<br>Bad: 这位记者批评了。 |
| 老板 → 裤子 | agent_animacy_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 老板观看过动画片了。<br>Bad: 裤子观看过动画片了。 |
| 走路 → 批评 | intransitive_no_obj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 他走路了。<br>Bad: 他批评了。 |
| 躺下 → 拥护 | intransitive_no_obj | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 你躺下了。<br>Bad: 你拥护了。 |
| 过来 → 支持 | intransitive_no_obj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 另外八位空姐过来了。<br>Bad: 另外八位空姐支持了。 |
| 郑大妈 → 充电器 | agent_animacy_passive | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 那张桌子被郑大妈清洗了。<br>Bad: 那张桌子被充电器清洗了。 |
| 音乐家 → 开瓶器 | agent_animacy_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 音乐家拉过大提琴了。<br>Bad: 开瓶器拉过大提琴了。 |
| 飞行员 → 热水器 | agent_animacy_subj | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 吴太太的飞行员把火车驾驶了。<br>Bad: 吴太太的热水器把火车驾驶了。 |
| 有点 → 专心 | agent_animacy_adv | 80 | 1.0000 | 0.5125 | +0.4875 | 0.0000 | Good: 三本书有点受潮了。<br>Bad: 三本书专心受潮了。 |
| bad deletes 它 | agent_deletion | 27 | 1.0000 | 0.5185 | +0.4815 | 0.0000 | Good: 杨大哥，这个秘密跟它没有关系。<br>Bad: 杨大哥，这个秘密跟没有关系。 |
| 可能 → 故意 | agent_animacy_adv | 5 | 0.6000 | 1.0000 | -0.4000 | 0.0000 | Good: 三部手账可能受潮了。<br>Bad: 三部手账故意受潮了。 |
| multiple edits: bad inserts 拿大象的; bad deletes 八头大象 | intransitive_double_obj | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 这位老板递给了宋女士八头大象。<br>Bad: 这位老板递给了拿大象的宋女士。 |
| multiple edits: bad inserts 拿大象的; bad deletes 几头大象 | intransitive_double_obj | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 另外九位吉他手卖给了张先生几头大象。<br>Bad: 另外九位吉他手卖给了拿大象的张先生。 |
| multiple edits: bad inserts 拿牛的; bad deletes 几头牛 | intransitive_double_obj | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这十个妹妹寄给了郑大妈几头牛。<br>Bad: 这十个妹妹寄给了拿牛的郑大妈。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 一条鱼 | intransitive_double_obj | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 小王的下属卖给了他们一条鱼。<br>Bad: 小王的下属卖给了拿鱼的他们。 |
| 屠宰 → 凝固 | agent_causative | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 何太太屠宰了牛。<br>Bad: 何太太凝固了牛。 |
| 检查 → 变化 | agent_causative | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 张三险些检查了胃。<br>Bad: 张三险些变化了胃。 |
| 预习 → 凝固 | agent_causative | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 他的下属险些预习了教材。<br>Bad: 他的下属险些凝固了教材。 |
| 驾驶 → 出现 | agent_causative | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 郑大妈的儿子驾驶了轮船。<br>Bad: 郑大妈的儿子出现了轮船。 |
| 创作 → 气化 | agent_causative | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 王先生的员工创作了漫画。<br>Bad: 王先生的员工气化了漫画。 |
| 制作 → 存在 | agent_causative | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 你们险些制作了动作片。<br>Bad: 你们险些存在了动作片。 |
| 她们 → 裙子 | agent_animacy_passive | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 这头牛被她们领养了。<br>Bad: 这头牛被裙子领养了。 |
| 捕捉 → 凝固 | agent_causative | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 她们捕捉了老虎。<br>Bad: 她们凝固了老虎。 |
| 消费者 → 饮料瓶 | agent_animacy_subj | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 消费者把电影观看了。<br>Bad: 饮料瓶把电影观看了。 |
| 演员 → 教材 | agent_animacy_subj | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 陈大姐的演员把大象捕捉了。<br>Bad: 陈大姐的教材把大象捕捉了。 |
| 老板 → 杯子 | agent_animacy_subj | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 老板跨越了沙漠。<br>Bad: 杯子跨越了沙漠。 |
| 领导 → 电影 | agent_animacy_subj | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 领导清洗过杯子了。<br>Bad: 电影清洗过杯子了。 |
| 驾驶 → 凝固 | agent_causative | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 杨大哥的老师差点儿驾驶了火车。<br>Bad: 杨大哥的老师差点儿凝固了火车。 |
| 可能 → 专心 | agent_animacy_adv | 72 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 七桶方便面可能变质了。<br>Bad: 七桶方便面专心变质了。 |
| 他们 → 电影 | agent_animacy_passive | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 那部电影被他们观看了。<br>Bad: 那部电影被电影观看了。 |
| 拍摄 → 凝固 | agent_causative | 4 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 她拍摄了电影。<br>Bad: 她凝固了电影。 |
| 演奏 → 凝固 | agent_causative | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 她们几乎演奏了奏鸣曲。<br>Bad: 她们几乎凝固了奏鸣曲。 |
| multiple edits: bad inserts 拿大象的; bad deletes 九头大象 | intransitive_double_obj | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 他的学生卖给了刘先生九头大象。<br>Bad: 他的学生卖给了拿大象的刘先生。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 两条鱼 | intransitive_double_obj | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 那九位下属借给了小明两条鱼。<br>Bad: 那九位下属借给了拿鱼的小明。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 九只鸭 | intransitive_double_obj | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 李四的上级借给了张婶九只鸭。<br>Bad: 李四的上级借给了拿鸭的张婶。 |
| 有点 → 努力 | agent_animacy_adv | 57 | 0.9123 | 0.7193 | +0.1930 | 0.0000 | Good: 五杯白酒有点变质了。<br>Bad: 五杯白酒努力变质了。 |
| 制作 → 融化 | agent_causative | 6 | 0.1667 | 0.3333 | -0.1667 | 0.0000 | Good: 这位打工人制作了电影。<br>Bad: 这位打工人融化了电影。 |
| bad deletes 他 | agent_deletion | 24 | 0.4583 | 0.2917 | +0.1667 | 0.0000 | Good: 王先生，那个事情跟他没有关系。<br>Bad: 王先生，那个事情跟没有关系。 |
| 可能 → 努力 | agent_animacy_adv | 81 | 0.9506 | 1.0000 | -0.0494 | 0.0000 | Good: 十几串香蕉可能腐烂了。<br>Bad: 十几串香蕉努力腐烂了。 |
| bad deletes 他们 | agent_deletion | 42 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太，那个东西和他们没有关系。<br>Bad: 李太太，那个东西和没有关系。 |
| bad deletes 我 | agent_deletion | 40 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥，那个事跟我没有关系。<br>Bad: 冯大哥，那个事跟没有关系。 |
| bad deletes 她们 | agent_deletion | 37 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐，那个东西跟她们没有关系。<br>Bad: 王小姐，那个东西跟没有关系。 |
| bad deletes 你们 | agent_deletion | 33 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈，这个信息和你们没有关系。<br>Bad: 周大妈，这个信息和没有关系。 |
| bad deletes 我们 | agent_deletion | 33 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨，这些信息跟我们没有关系。<br>Bad: 王姨，这些信息跟没有关系。 |
| bad deletes 你 | agent_deletion | 31 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐，那些新闻和你没有关系。<br>Bad: 王小姐，那些新闻和没有关系。 |
| 领养 → 蒸发 | agent_causative | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外七个打工人领养了小猫。<br>Bad: 另外七个打工人蒸发了小猫。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 十几条鱼 | intransitive_double_obj | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶的领导卖给了我们十几条鱼。<br>Bad: 张婶的领导卖给了拿鱼的我们。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 许多条鱼 | intransitive_double_obj | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一个吉他手送给了王大娘许多条鱼。<br>Bad: 这一个吉他手送给了拿鱼的王大娘。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 七只鸡 | intransitive_double_obj | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那一位打工人递给了王大娘七只鸡。<br>Bad: 那一位打工人递给了拿鸡的王大娘。 |
| 屠宰 → 变化 | agent_causative | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈的弟弟屠宰了牛。<br>Bad: 郑大妈的弟弟变化了牛。 |
| 捕捉 → 消失 | agent_causative | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们差点儿捕捉了蛇。<br>Bad: 我们差点儿消失了蛇。 |
| 清洗 → 变化 | agent_causative | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们几乎清洗了杯子。<br>Bad: 我们几乎变化了杯子。 |
| 他们 → 衣服 | agent_animacy_passive | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六头大象被他们麻醉了。<br>Bad: 这六头大象被衣服麻醉了。 |
| 你们 → 教材 | agent_animacy_passive | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这桶方便面被你们吃了。<br>Bad: 这桶方便面被教材吃了。 |
| 创作 → 消失 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生险些创作了小说。<br>Bad: 李先生险些消失了小说。 |
| 制作 → 变化 | agent_causative | 4 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 这个顾客制作了动作片。<br>Bad: 这个顾客变化了动作片。 |
| 她们 → 教材 | agent_animacy_passive | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那张桌子被她们搬了。<br>Bad: 那张桌子被教材搬了。 |
| 我们 → 教材 | agent_animacy_passive | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位下属被我们安慰了。<br>Bad: 那位下属被教材安慰了。 |
| 拍摄 → 出现 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生差点儿拍摄了电影。<br>Bad: 李先生差点儿出现了电影。 |
| 检查 → 消失 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我检查了头。<br>Bad: 我消失了头。 |
| 清蒸 → 凝固 | agent_causative | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们清蒸了鸭。<br>Bad: 她们凝固了鸭。 |
| 演奏 → 存在 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨险些演奏了狂想曲。<br>Bad: 王姨险些存在了狂想曲。 |
| 预习 → 变化 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九位工人几乎预习了教材。<br>Bad: 这九位工人几乎变化了教材。 |
| 预习 → 存在 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的老师预习了教材。<br>Bad: 他的老师存在了教材。 |
| 领养 → 凝固 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个打工人领养了小狗。<br>Bad: 那个打工人凝固了小狗。 |
| 领养 → 变化 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那四位消费者几乎领养了小狗。<br>Bad: 那四位消费者几乎变化了小狗。 |
| 领养 → 融化 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太领养了小猫。<br>Bad: 李太太融化了小猫。 |
| 驾驶 → 蒸发 | agent_causative | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我驾驶了卡车。<br>Bad: 我蒸发了卡车。 |
| multiple edits: bad inserts 拿大象的; bad deletes 三头大象 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个姐姐送给了何太太三头大象。<br>Bad: 那个姐姐送给了拿大象的何太太。 |
| multiple edits: bad inserts 拿大象的; bad deletes 两头大象 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的爸爸寄给了周大妈两头大象。<br>Bad: 她们的爸爸寄给了拿大象的周大妈。 |
| multiple edits: bad inserts 拿大象的; bad deletes 好几头大象 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位服务员送给了她们好几头大象。<br>Bad: 这位服务员送给了拿大象的她们。 |
| multiple edits: bad inserts 拿父亲的; bad deletes 四位父亲 | intransitive_double_obj | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 张先生的领导买给了张三四位父亲。<br>Bad: 张先生的领导买给了拿父亲的张三。 |
| multiple edits: bad inserts 拿牛的; bad deletes 好几头牛 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那七位钢琴家送给了李四好几头牛。<br>Bad: 那七位钢琴家送给了拿牛的李四。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 三条鱼 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那六个弟弟寄给了我们三条鱼。<br>Bad: 那六个弟弟寄给了拿鱼的我们。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 十条鱼 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个小孩买给了你十条鱼。<br>Bad: 那个小孩买给了拿鱼的你。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 好几十条鱼 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个朋友寄给了他们好几十条鱼。<br>Bad: 这个朋友寄给了拿鱼的他们。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 两只鸡 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐的员工递给了他两只鸡。<br>Bad: 徐小姐的员工递给了拿鸡的他。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 五只鸡 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三的领导买给了我们五只鸡。<br>Bad: 张三的领导买给了拿鸡的我们。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 四只鸡 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的女儿卖给了她们四只鸡。<br>Bad: 我们的女儿卖给了拿鸡的她们。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 四只鸭 | intransitive_double_obj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五个音乐家卖给了你四只鸭。<br>Bad: 这五个音乐家卖给了拿鸭的你。 |
| 你们 → 杯子 | agent_animacy_passive | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那只脚被你们检查了。<br>Bad: 那只脚被杯子检查了。 |
| 兄弟 → 手套 | agent_animacy_subj | 3 | 0.3333 | 0.3333 | +0.0000 | 0.0000 | Good: 兄弟煮过鱼了。<br>Bad: 手套煮过鱼了。 |
| 包扎 → 变化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的同事包扎了肚子。<br>Bad: 他的同事变化了肚子。 |
| 包扎 → 消失 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她险些包扎了脚。<br>Bad: 她险些消失了脚。 |
| 包扎 → 蒸发 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八位母亲险些包扎了手。<br>Bad: 这八位母亲险些蒸发了手。 |
| 她们 → 小说 | agent_animacy_passive | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十本教材被她们盖了。<br>Bad: 另外十本教材被小说盖了。 |
| 小明 → 教材 | agent_animacy_passive | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三串香蕉被小明吃了。<br>Bad: 这三串香蕉被教材吃了。 |
| 小王 → 衣服 | agent_animacy_passive | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这部小说被小王观看了。<br>Bad: 这部小说被衣服观看了。 |
| 小王 → 袜子 | agent_animacy_passive | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外三只鸡被小王捕捉了。<br>Bad: 另外三只鸡被袜子捕捉了。 |
| 屠宰 → 出现 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐屠宰了牛。<br>Bad: 徐小姐出现了牛。 |
| 工人 → 手套 | agent_animacy_subj | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 工人唱了京剧。<br>Bad: 手套唱了京剧。 |
| 我们 → 杯子 | agent_animacy_passive | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外三只鸡被我们清蒸了。<br>Bad: 另外三只鸡被杯子清蒸了。 |
| 我们 → 裤子 | agent_animacy_passive | 3 | 0.3333 | 0.3333 | +0.0000 | 0.0000 | Good: 这四头大象被我们领养了。<br>Bad: 这四头大象被裤子领养了。 |
| 打断 → 凝固 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太的姐姐打断了脚。<br>Bad: 李太太的姐姐凝固了脚。 |
| 打断 → 出现 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个儿子差点儿打断了腿。<br>Bad: 那个儿子差点儿出现了腿。 |
| 打断 → 变化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生打断了鼻子。<br>Bad: 张先生变化了鼻子。 |
| 打断 → 存在 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的女儿打断了脚。<br>Bad: 他的女儿存在了脚。 |
| 打断 → 气化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个女儿几乎打断了鼻子。<br>Bad: 这个女儿几乎气化了鼻子。 |
| 清洗 → 出现 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个钢琴家清洗了杯子。<br>Bad: 这个钢琴家出现了杯子。 |
| 清洗 → 存在 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那十位老板险些清洗了杯子。<br>Bad: 那十位老板险些存在了杯子。 |
| 演奏 → 气化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个领导险些演奏了歌曲。<br>Bad: 这个领导险些气化了歌曲。 |
| 舞者 → 教材 | agent_animacy_subj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨的舞者把鸡捕捉了。<br>Bad: 王姨的教材把鸡捕捉了。 |
| 观看 → 变化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那七位司机观看了电影。<br>Bad: 那七位司机变化了电影。 |
| 观看 → 气化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太差点儿观看了动画片。<br>Bad: 李太太差点儿气化了动画片。 |
| 观看 → 蒸发 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位上级观看了电视剧。<br>Bad: 这位上级蒸发了电视剧。 |
| 记者 → 电影 | agent_animacy_subj | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 记者把鱼炖了。<br>Bad: 电影把鱼炖了。 |
| 记者 → 袜子 | agent_animacy_subj | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三的记者把腿打断了。<br>Bad: 张三的袜子把腿打断了。 |
| 跨越 → 变化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个顾客跨越了海洋。<br>Bad: 那个顾客变化了海洋。 |
| 预习 → 出现 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她预习了教材。<br>Bad: 她出现了教材。 |
| 预习 → 消失 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个领导差点儿预习了教材。<br>Bad: 这个领导差点儿消失了教材。 |
| 驾驶 → 存在 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生险些驾驶了飞机。<br>Bad: 刘先生险些存在了飞机。 |
| 麻醉 → 出现 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位钢琴家差点儿麻醉了老虎。<br>Bad: 这位钢琴家差点儿出现了老虎。 |
| 麻醉 → 变化 | agent_causative | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们差点儿麻醉了老虎。<br>Bad: 他们差点儿变化了老虎。 |
| multiple edits: bad deletes 清; bad inserts 发 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐的姐妹清蒸了鸭。<br>Bad: 徐小姐的姐妹蒸发了鸭。 |
| multiple edits: bad inserts 拿大象的; bad deletes 七头大象 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个演奏员卖给了他们七头大象。<br>Bad: 那个演奏员卖给了拿大象的他们。 |
| multiple edits: bad inserts 拿小狗的; bad deletes 好几十条小狗 | intransitive_double_obj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 张婶的姐妹寄给了你们好几十条小狗。<br>Bad: 张婶的姐妹寄给了拿小狗的你们。 |
| multiple edits: bad inserts 拿牛的; bad deletes 好几百头牛 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的上级借给了王先生好几百头牛。<br>Bad: 她的上级借给了拿牛的王先生。 |
| multiple edits: bad inserts 拿蛇的; bad deletes 五条蛇 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三位打工人送给了吴太太五条蛇。<br>Bad: 这三位打工人送给了拿蛇的吴太太。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 七条鱼 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个演员送给了你们七条鱼。<br>Bad: 那个演员送给了拿鱼的你们。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 五条鱼 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个顾客买给了他五条鱼。<br>Bad: 这个顾客买给了拿鱼的他。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 八条鱼 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个演员递给了你们八条鱼。<br>Bad: 那个演员递给了拿鱼的你们。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 几条鱼 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位消费者卖给了徐小姐几条鱼。<br>Bad: 那位消费者卖给了拿鱼的徐小姐。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 非常多条鱼 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个弟弟买给了你们非常多条鱼。<br>Bad: 那个弟弟买给了拿鱼的你们。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 七只鸭 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位服务员借给了他们七只鸭。<br>Bad: 这位服务员借给了拿鸭的他们。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 十几只鸭 | intransitive_double_obj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那位学生寄给了你十几只鸭。<br>Bad: 那位学生寄给了拿鸭的你。 |
| multiple edits: 你八个 -> 拿; bad inserts 的你 | intransitive_double_obj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的姐妹卖给了你八个朋友。<br>Bad: 李太太的姐妹卖给了拿朋友的你。 |
| multiple edits: 你几位 -> 拿; bad inserts 的你 | intransitive_double_obj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那九位钢琴家寄给了你几位顾客。<br>Bad: 那九位钢琴家寄给了拿顾客的你。 |
| multiple edits: 我许多位 -> 拿; bad inserts 的我 | intransitive_double_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太的同事送给了我许多位空姐。<br>Bad: 吴太太的同事送给了拿空姐的我。 |
| 上级 → 袜子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的上级拉了大提琴。<br>Bad: 李太太的袜子拉了大提琴。 |
| 下属 → 袜子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五的下属把火车开了。<br>Bad: 王五的袜子把火车开了。 |
| 下属 → 裙子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 下属把橙汁喝了。<br>Bad: 裙子把橙汁喝了。 |
| 他们 → 手套 | agent_animacy_passive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这本漫画被他们创作了。<br>Bad: 这本漫画被手套创作了。 |
| 他们 → 裤子 | agent_animacy_passive | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 这十本教材被他们喝了。<br>Bad: 这十本教材被裤子喝了。 |
| 你们 → 小说 | agent_animacy_passive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那杯红酒被你们喝了。<br>Bad: 那杯红酒被小说喝了。 |
| 你们 → 衣服 | agent_animacy_passive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外七串香蕉被你们买了。<br>Bad: 另外七串香蕉被衣服买了。 |
| 你们 → 裤子 | agent_animacy_passive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这头大象被你们领养了。<br>Bad: 这头大象被裤子领养了。 |
| 制作 → 消失 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王制作了视频。<br>Bad: 小王消失了视频。 |
| 包扎 → 存在 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的下属差点儿包扎了脚。<br>Bad: 她的下属差点儿存在了脚。 |
| 包扎 → 气化 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐险些包扎了腿。<br>Bad: 陈大姐险些气化了腿。 |
| 司机 → 衣服 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 我的司机驾驶过货车了。<br>Bad: 我的衣服驾驶过货车了。 |
| 女儿 → 杯子 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 女儿拉了小提琴。<br>Bad: 杯子拉了小提琴。 |
| 她们 → 作业 | agent_animacy_passive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五张桌子被她们搬了。<br>Bad: 这五张桌子被作业搬了。 |
| 姐姐 → 杯子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我的姐姐检查过耳朵了。<br>Bad: 我的杯子检查过耳朵了。 |
| 屠宰 → 气化 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷的老板几乎屠宰了牛。<br>Bad: 胡大爷的老板几乎气化了牛。 |
| 屠宰 → 蒸发 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生屠宰了牛。<br>Bad: 王先生蒸发了牛。 |
| 屠宰 → 融化 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她屠宰了牛。<br>Bad: 她融化了牛。 |
| 工人 → 教材 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 工人把录像带看了。<br>Bad: 教材把录像带看了。 |
| 工人 → 裙子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们的工人预习了教材。<br>Bad: 她们的裙子预习了教材。 |
| 弟弟 → 手套 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你的弟弟煮过鱼了。<br>Bad: 你的手套煮过鱼了。 |
| 张三 → 橙汁 | agent_animacy_passive | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那八杯白酒被张三买了。<br>Bad: 那八杯白酒被橙汁买了。 |
| 张先生 → 饮料瓶 | agent_animacy_passive | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那杯红茶被张先生喝了。<br>Bad: 那杯红茶被饮料瓶喝了。 |
| 张夫人 → 开瓶器 | agent_animacy_passive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那把椅子被张夫人喝了。<br>Bad: 那把椅子被开瓶器喝了。 |
| 微笑 → 捕捉 | intransitive_no_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太微笑了。<br>Bad: 吴太太捕捉了。 |
| 打断 → 消失 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人打断了腿。<br>Bad: 张夫人消失了腿。 |
| 打架 → 厌恶 | intransitive_no_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位音乐家打架了。<br>Bad: 这位音乐家厌恶了。 |
| 捕捉 → 蒸发 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外九个吉他手捕捉了鸡。<br>Bad: 另外九个吉他手蒸发了鸡。 |
| 有点 → 故意 | agent_animacy_adv | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 十片面包有点变质了。<br>Bad: 十片面包故意变质了。 |
| 李四 → 手套 | agent_animacy_passive | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那八头大象被李四麻醉了。<br>Bad: 那八头大象被手套麻醉了。 |
| 李四 → 杯子 | agent_animacy_passive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外九部漫画被李四观看了。<br>Bad: 另外九部漫画被杯子观看了。 |
| 母亲 → 红酒 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 母亲观看了动作片。<br>Bad: 红酒观看了动作片。 |
| 消费者 → 开瓶器 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 消费者把视频制作了。<br>Bad: 开瓶器把视频制作了。 |
| 清洗 → 消失 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的下属清洗了杯子。<br>Bad: 他们的下属消失了杯子。 |
| 清蒸 → 变化 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我清蒸了鸭。<br>Bad: 我变化了鸭。 |
| 溜走 → 诽谤 | intransitive_no_obj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们溜走了。<br>Bad: 他们诽谤了。 |
| 演员 → 手套 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 周大妈的演员清洗过杯子了。<br>Bad: 周大妈的手套清洗过杯子了。 |
| 演员 → 杯子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他的演员麻醉过老虎了。<br>Bad: 他的杯子麻醉过老虎了。 |
| 演员 → 袜子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 演员领养了小猫。<br>Bad: 袜子领养了小猫。 |
| 演奏 → 出现 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个儿子演奏了华尔兹。<br>Bad: 这个儿子出现了华尔兹。 |
| 演奏 → 消失 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位演奏员演奏了歌曲。<br>Bad: 这位演奏员消失了歌曲。 |
| 演奏 → 蒸发 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐的姐姐差点儿演奏了歌曲。<br>Bad: 王小姐的姐姐差点儿蒸发了歌曲。 |
| 爆炒 → 消失 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十位老师差点儿爆炒了鸭。<br>Bad: 另外十位老师差点儿消失了鸭。 |
| 王大娘 → 充电器 | agent_animacy_passive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这头大象被王大娘清蒸了。<br>Bad: 这头大象被充电器清蒸了。 |
| 玩耍 → 反感 | intransitive_no_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外两位演员玩耍了。<br>Bad: 另外两位演员反感了。 |
| 罪犯 → 袜子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人的罪犯把鼻子打断了。<br>Bad: 张夫人的袜子把鼻子打断了。 |
| 老师 → 衣服 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 我的老师开了飞机。<br>Bad: 我的衣服开了飞机。 |
| 胡大爷 → 充电器 | agent_animacy_passive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那头大象被胡大爷捕捉了。<br>Bad: 那头大象被充电器捕捉了。 |
| 胡大爷 → 开瓶器 | agent_animacy_passive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外一把椅子被胡大爷吃了。<br>Bad: 另外一把椅子被开瓶器吃了。 |
| 胡大爷 → 热水器 | agent_animacy_passive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五只鸡被胡大爷煮了。<br>Bad: 这五只鸡被热水器煮了。 |
| 记者 → 椅子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的记者吃过方便面了。<br>Bad: 他们的椅子吃过方便面了。 |
| 走 → 喝 | intransitive_no_obj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生走了。<br>Bad: 李先生喝了。 |
| 走路 → 支持 | intransitive_no_obj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐走路了。<br>Bad: 徐小姐支持了。 |
| 赵大爷 → 电视机 | agent_animacy_passive | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那部电影被赵大爷观看了。<br>Bad: 那部电影被电视机观看了。 |
| 跨越 → 消失 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一个姐姐几乎跨越了海洋。<br>Bad: 另外一个姐姐几乎消失了海洋。 |
| 音乐家 → 饮料瓶 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 音乐家拉了大提琴。<br>Bad: 饮料瓶拉了大提琴。 |
| 顾客 → 手套 | agent_animacy_subj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 顾客弹了古筝。<br>Bad: 手套弹了古筝。 |
| 预习 → 蒸发 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十位父亲差点儿预习了教材。<br>Bad: 另外十位父亲差点儿蒸发了教材。 |
| 领养 → 存在 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的母亲差点儿领养了小狗。<br>Bad: 你的母亲差点儿存在了小狗。 |
| 领养 → 气化 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你领养了小猫。<br>Bad: 你气化了小猫。 |
| 领导 → 桌子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘的领导把沙漠跨越了。<br>Bad: 王大娘的桌子把沙漠跨越了。 |
| 领导 → 椅子 | agent_animacy_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 领导屠宰了牛。<br>Bad: 椅子屠宰了牛。 |
| 驾驶 → 变化 | agent_causative | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他驾驶了飞机。<br>Bad: 他变化了飞机。 |
| 麻醉 → 融化 | agent_causative | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位吉他手麻醉了大象。<br>Bad: 那位吉他手融化了大象。 |
| multiple edits: bad inserts 拿上级的; bad deletes 九个上级 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥的弟弟卖给了你们九个上级。<br>Bad: 冯大哥的弟弟卖给了拿上级的你们。 |
| multiple edits: bad inserts 拿上级的; bad deletes 八个上级 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位吉他手送给了冯大哥八个上级。<br>Bad: 这位吉他手送给了拿上级的冯大哥。 |
| multiple edits: bad inserts 拿上级的; bad deletes 好几个上级 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个顾客送给了何太太好几个上级。<br>Bad: 这个顾客送给了拿上级的何太太。 |
| multiple edits: bad inserts 拿下属的; bad deletes 九个下属 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位领导寄给了我们九个下属。<br>Bad: 那位领导寄给了拿下属的我们。 |
| multiple edits: bad inserts 拿下属的; bad deletes 九位下属 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外五位司机递给了她们九位下属。<br>Bad: 另外五位司机递给了拿下属的她们。 |
| multiple edits: bad inserts 拿儿子的; bad deletes 六个儿子 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐的儿子寄给了王大娘六个儿子。<br>Bad: 徐小姐的儿子寄给了拿儿子的王大娘。 |
| multiple edits: bad inserts 拿儿子的; bad deletes 非常多个儿子 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外七个演员借给了张先生非常多个儿子。<br>Bad: 另外七个演员借给了拿儿子的张先生。 |
| multiple edits: bad inserts 拿司机的; bad deletes 几位司机 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个演奏员寄给了王姨几位司机。<br>Bad: 这个演奏员寄给了拿司机的王姨。 |
| multiple edits: bad inserts 拿司机的; bad deletes 十几位司机 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨的下属借给了陈大姐十几位司机。<br>Bad: 王姨的下属借给了拿司机的陈大姐。 |
| multiple edits: bad inserts 拿吉他手的; bad deletes 九个吉他手 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外两个妹妹卖给了何太太九个吉他手。<br>Bad: 另外两个妹妹卖给了拿吉他手的何太太。 |
| multiple edits: bad inserts 拿吉他手的; bad deletes 好几十个吉他手 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六个哥哥借给了吴太太好几十个吉他手。<br>Bad: 这六个哥哥借给了拿吉他手的吴太太。 |
| multiple edits: bad inserts 拿吉他手的; bad deletes 许多个吉他手 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我的上级买给了王先生许多个吉他手。<br>Bad: 我的上级买给了拿吉他手的王先生。 |
| multiple edits: bad inserts 拿同事的; bad deletes 几位同事 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太的领导卖给了你们几位同事。<br>Bad: 何太太的领导卖给了拿同事的你们。 |
| multiple edits: bad inserts 拿员工的; bad deletes 九位员工 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷的儿子递给了杨大哥九位员工。<br>Bad: 赵大爷的儿子递给了拿员工的杨大哥。 |
| multiple edits: bad inserts 拿员工的; bad deletes 好几位员工 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个上级递给了张夫人好几位员工。<br>Bad: 这个上级递给了拿员工的张夫人。 |
| multiple edits: bad inserts 拿员工的; bad deletes 许多位员工 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五的学生借给了你们许多位员工。<br>Bad: 王五的学生借给了拿员工的你们。 |
| multiple edits: bad inserts 拿哥哥的; bad deletes 四个哥哥 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这两位下属递给了赵大爷四个哥哥。<br>Bad: 这两位下属递给了拿哥哥的赵大爷。 |
| multiple edits: bad inserts 拿哥哥的; bad deletes 好几个哥哥 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的女儿递给了吴太太好几个哥哥。<br>Bad: 你们的女儿递给了拿哥哥的吴太太。 |
| multiple edits: bad inserts 拿大象的; bad deletes 好几十头大象 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的姐妹寄给了李先生好几十头大象。<br>Bad: 她们的姐妹寄给了拿大象的李先生。 |
| multiple edits: bad inserts 拿大象的; bad deletes 好几百头大象 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个领导买给了你们好几百头大象。<br>Bad: 那个领导买给了拿大象的你们。 |
| multiple edits: bad inserts 拿奴隶的; bad deletes 几个奴隶 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个女儿递给了周大妈几个奴隶。<br>Bad: 这个女儿递给了拿奴隶的周大妈。 |
| multiple edits: bad inserts 拿奴隶的; bad deletes 非常多个奴隶 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的员工递给了王大娘非常多个奴隶。<br>Bad: 他们的员工递给了拿奴隶的王大娘。 |
| multiple edits: bad inserts 拿妹妹的; bad deletes 九个妹妹 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生的父亲递给了张先生九个妹妹。<br>Bad: 李先生的父亲递给了拿妹妹的张先生。 |
| multiple edits: bad inserts 拿妹妹的; bad deletes 几个妹妹 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的领导买给了你们几个妹妹。<br>Bad: 李太太的领导买给了拿妹妹的你们。 |
| multiple edits: bad inserts 拿妹妹的; bad deletes 好几个妹妹 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一个顾客借给了你们好几个妹妹。<br>Bad: 另外一个顾客借给了拿妹妹的你们。 |
| multiple edits: bad inserts 拿姐姐的; bad deletes 好几百个姐姐 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个下属借给了小王好几百个姐姐。<br>Bad: 这个下属借给了拿姐姐的小王。 |
| multiple edits: bad inserts 拿学生的; bad deletes 八位学生 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王的朋友递给了吴太太八位学生。<br>Bad: 小王的朋友递给了拿学生的吴太太。 |
| multiple edits: bad inserts 拿学生的; bad deletes 六位学生 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位领导借给了王大娘六位学生。<br>Bad: 那位领导借给了拿学生的王大娘。 |
| multiple edits: bad inserts 拿学生的; bad deletes 非常多位学生 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个音乐家买给了我们非常多位学生。<br>Bad: 那个音乐家买给了拿学生的我们。 |
| multiple edits: bad inserts 拿小孩的; bad deletes 五个小孩 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他的姐妹递给了吴太太五个小孩。<br>Bad: 他的姐妹递给了拿小孩的吴太太。 |
| multiple edits: bad inserts 拿小狗的; bad deletes 三条小狗 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这十位上级送给了杨大哥三条小狗。<br>Bad: 这十位上级送给了拿小狗的杨大哥。 |
| multiple edits: bad inserts 拿小狗的; bad deletes 十几条小狗 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九个记者借给了胡大爷十几条小狗。<br>Bad: 这九个记者借给了拿小狗的胡大爷。 |
| multiple edits: bad inserts 拿小猫的; bad deletes 一只小猫 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生的老板送给了他们一只小猫。<br>Bad: 王先生的老板送给了拿小猫的他们。 |
| multiple edits: bad inserts 拿小猫的; bad deletes 六只小猫 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的员工递给了王姨六只小猫。<br>Bad: 她的员工递给了拿小猫的王姨。 |
| multiple edits: bad inserts 拿工人的; bad deletes 五位工人 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那个儿子寄给了她们五位工人。<br>Bad: 那个儿子寄给了拿工人的她们。 |
| multiple edits: bad inserts 拿工人的; bad deletes 十个工人 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位演员递给了她们十个工人。<br>Bad: 这位演员递给了拿工人的她们。 |
| multiple edits: bad inserts 拿工人的; bad deletes 好几位工人 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九个工人卖给了我们好几位工人。<br>Bad: 这九个工人卖给了拿工人的我们。 |
| multiple edits: bad inserts 拿工人的; bad deletes 许多位工人 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位司机寄给了刘先生许多位工人。<br>Bad: 这位司机寄给了拿工人的刘先生。 |
| multiple edits: bad inserts 拿弟弟的; bad deletes 五个弟弟 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的儿子买给了他们五个弟弟。<br>Bad: 你的儿子买给了拿弟弟的他们。 |
| multiple edits: bad inserts 拿打工人的; bad deletes 好几位打工人 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那十位音乐家借给了张先生好几位打工人。<br>Bad: 那十位音乐家借给了拿打工人的张先生。 |
| multiple edits: bad inserts 拿朋友的; bad deletes 九个朋友 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个上级借给了王小姐九个朋友。<br>Bad: 这个上级借给了拿朋友的王小姐。 |
| multiple edits: bad inserts 拿消费者的; bad deletes 九位消费者 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个领导送给了赵大爷九位消费者。<br>Bad: 这个领导送给了拿消费者的赵大爷。 |
| multiple edits: bad inserts 拿消费者的; bad deletes 六个消费者 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们的领导递给了冯大哥六个消费者。<br>Bad: 我们的领导递给了拿消费者的冯大哥。 |
| multiple edits: bad inserts 拿演员的; bad deletes 八位演员 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那三位空姐买给了王小姐八位演员。<br>Bad: 那三位空姐买给了拿演员的王小姐。 |
| multiple edits: bad inserts 拿演员的; bad deletes 许多个演员 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八位下属递给了王小姐许多个演员。<br>Bad: 这八位下属递给了拿演员的王小姐。 |
| multiple edits: bad inserts 拿演奏员的; bad deletes 一位演奏员 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶的老板卖给了郑大妈一位演奏员。<br>Bad: 张婶的老板卖给了拿演奏员的郑大妈。 |
| multiple edits: bad inserts 拿演奏员的; bad deletes 三个演奏员 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位打工人卖给了周大妈三个演奏员。<br>Bad: 那位打工人卖给了拿演奏员的周大妈。 |
| multiple edits: bad inserts 拿父亲的; bad deletes 两位父亲 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个领导买给了吴太太两位父亲。<br>Bad: 这个领导买给了拿父亲的吴太太。 |
| multiple edits: bad inserts 拿父亲的; bad deletes 许多位父亲 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的姐姐借给了郑大妈许多位父亲。<br>Bad: 他的姐姐借给了拿父亲的郑大妈。 |
| multiple edits: bad inserts 拿牛的; bad deletes 一头牛 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位老师借给了她一头牛。<br>Bad: 这位老师借给了拿牛的她。 |
| multiple edits: bad inserts 拿牛的; bad deletes 七头牛 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那两个儿子卖给了你们七头牛。<br>Bad: 那两个儿子卖给了拿牛的你们。 |
| multiple edits: bad inserts 拿牛的; bad deletes 三头牛 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的同事卖给了张婶三头牛。<br>Bad: 我的同事卖给了拿牛的张婶。 |
| multiple edits: bad inserts 拿牛的; bad deletes 五头牛 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王的儿子买给了她五头牛。<br>Bad: 小王的儿子买给了拿牛的她。 |
| multiple edits: bad inserts 拿牛的; bad deletes 四头牛 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的上级递给了我四头牛。<br>Bad: 他的上级递给了拿牛的我。 |
| multiple edits: bad inserts 拿牛的; bad deletes 许多头牛 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的儿子寄给了她许多头牛。<br>Bad: 你的儿子寄给了拿牛的她。 |
| multiple edits: bad inserts 拿罪犯的; bad deletes 好几个罪犯 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的爸爸借给了他们好几个罪犯。<br>Bad: 我的爸爸借给了拿罪犯的他们。 |
| multiple edits: bad inserts 拿罪犯的; bad deletes 好几百个罪犯 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥的员工借给了他们好几百个罪犯。<br>Bad: 冯大哥的员工借给了拿罪犯的他们。 |
| multiple edits: bad inserts 拿老师的; bad deletes 六位老师 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的兄弟借给了她们六位老师。<br>Bad: 你们的兄弟借给了拿老师的她们。 |
| multiple edits: bad inserts 拿老板的; bad deletes 六个老板 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈的妈妈借给了杨大哥六个老板。<br>Bad: 周大妈的妈妈借给了拿老板的杨大哥。 |
| multiple edits: bad inserts 拿老板的; bad deletes 几个老板 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人的爸爸递给了周大妈几个老板。<br>Bad: 张夫人的爸爸递给了拿老板的周大妈。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 九只老虎 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这八位老板买给了周大妈九只老虎。<br>Bad: 这八位老板买给了拿老虎的周大妈。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 几只老虎 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外五个钢琴家借给了郑大妈几只老虎。<br>Bad: 另外五个钢琴家借给了拿老虎的郑大妈。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 好几十只老虎 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那六个舞者递给了他们好几十只老虎。<br>Bad: 那六个舞者递给了拿老虎的他们。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 好几百只老虎 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位演员寄给了张婶好几百只老虎。<br>Bad: 这位演员寄给了拿老虎的张婶。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 许多只老虎 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位母亲借给了你们许多只老虎。<br>Bad: 那位母亲借给了拿老虎的你们。 |
| multiple edits: bad inserts 拿老虎的; bad deletes 非常多只老虎 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太的姐姐卖给了小王非常多只老虎。<br>Bad: 何太太的姐姐卖给了拿老虎的小王。 |
| multiple edits: bad inserts 拿舞者的; bad deletes 九个舞者 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王的老板借给了小明九个舞者。<br>Bad: 小王的老板借给了拿舞者的小明。 |
| multiple edits: bad inserts 拿舞者的; bad deletes 六位舞者 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五位记者递给了王大娘六位舞者。<br>Bad: 这五位记者递给了拿舞者的王大娘。 |
| multiple edits: bad inserts 拿舞者的; bad deletes 好几位舞者 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐的父亲送给了他们好几位舞者。<br>Bad: 王小姐的父亲送给了拿舞者的他们。 |
| multiple edits: bad inserts 拿舞者的; bad deletes 许多位舞者 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那九位司机买给了宋女士许多位舞者。<br>Bad: 那九位司机买给了拿舞者的宋女士。 |
| multiple edits: bad inserts 拿舞者的; bad deletes 非常多位舞者 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的领导借给了王小姐非常多位舞者。<br>Bad: 你们的领导借给了拿舞者的王小姐。 |
| multiple edits: bad inserts 拿蛇的; bad deletes 九条蛇 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈的员工卖给了你们九条蛇。<br>Bad: 郑大妈的员工卖给了拿蛇的你们。 |
| multiple edits: bad inserts 拿蛇的; bad deletes 十几条蛇 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七位上级寄给了王先生十几条蛇。<br>Bad: 这七位上级寄给了拿蛇的王先生。 |
| multiple edits: bad inserts 拿蛇的; bad deletes 好几十条蛇 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生的母亲借给了张夫人好几十条蛇。<br>Bad: 刘先生的母亲借给了拿蛇的张夫人。 |
| multiple edits: bad inserts 拿蛇的; bad deletes 非常多条蛇 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们的同事买给了胡大爷非常多条蛇。<br>Bad: 她们的同事买给了拿蛇的胡大爷。 |
| multiple edits: bad inserts 拿记者的; bad deletes 两位记者 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那九位司机买给了冯大哥两位记者。<br>Bad: 那九位司机买给了拿记者的冯大哥。 |
| multiple edits: bad inserts 拿记者的; bad deletes 八个记者 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个弟弟递给了冯大哥八个记者。<br>Bad: 这个弟弟递给了拿记者的冯大哥。 |
| multiple edits: bad inserts 拿钢琴家的; bad deletes 好几个钢琴家 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥的老师借给了何太太好几个钢琴家。<br>Bad: 冯大哥的老师借给了拿钢琴家的何太太。 |
| multiple edits: bad inserts 拿音乐家的; bad deletes 一位音乐家 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这一位顾客买给了张先生一位音乐家。<br>Bad: 这一位顾客买给了拿音乐家的张先生。 |
| multiple edits: bad inserts 拿音乐家的; bad deletes 好几百个音乐家 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个服务员送给了吴太太好几百个音乐家。<br>Bad: 这个服务员送给了拿音乐家的吴太太。 |
| multiple edits: bad inserts 拿顾客的; bad deletes 七个顾客 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这三个奴隶递给了张婶七个顾客。<br>Bad: 这三个奴隶递给了拿顾客的张婶。 |
| multiple edits: bad inserts 拿顾客的; bad deletes 四个顾客 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位顾客递给了他们四个顾客。<br>Bad: 那位顾客递给了拿顾客的他们。 |
| multiple edits: bad inserts 拿领导的; bad deletes 九位领导 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的同事买给了他们九位领导。<br>Bad: 我们的同事买给了拿领导的他们。 |
| multiple edits: bad inserts 拿领导的; bad deletes 好几十个领导 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷的同事卖给了王小姐好几十个领导。<br>Bad: 赵大爷的同事卖给了拿领导的王小姐。 |
| multiple edits: bad inserts 拿领导的; bad deletes 非常多位领导 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这十位工人买给了李太太非常多位领导。<br>Bad: 这十位工人买给了拿领导的李太太。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 九条鱼 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士的老板买给了何太太九条鱼。<br>Bad: 宋女士的老板买给了拿鱼的何太太。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 六条鱼 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六位老师借给了你六条鱼。<br>Bad: 这六位老师借给了拿鱼的你。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 好几条鱼 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个记者递给了你们好几条鱼。<br>Bad: 那个记者递给了拿鱼的你们。 |
| multiple edits: bad inserts 拿鱼的; bad deletes 好几百条鱼 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个下属借给了我好几百条鱼。<br>Bad: 这个下属借给了拿鱼的我。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 九只鸡 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的姐妹送给了郑大妈九只鸡。<br>Bad: 我的姐妹送给了拿鸡的郑大妈。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 好几十只鸡 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位空姐递给了周大妈好几十只鸡。<br>Bad: 那位空姐递给了拿鸡的周大妈。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 好几百只鸡 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐的儿子借给了你们好几百只鸡。<br>Bad: 陈大姐的儿子借给了拿鸡的你们。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 许多只鸡 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生的弟弟借给了我许多只鸡。<br>Bad: 李先生的弟弟借给了拿鸡的我。 |
| multiple edits: bad inserts 拿鸡的; bad deletes 非常多只鸡 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个记者寄给了杨大哥非常多只鸡。<br>Bad: 那个记者寄给了拿鸡的杨大哥。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 几只鸭 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明的老师卖给了李四几只鸭。<br>Bad: 小明的老师卖给了拿鸭的李四。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 好几十只鸭 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这两个姐姐送给了我好几十只鸭。<br>Bad: 这两个姐姐送给了拿鸭的我。 |
| multiple edits: bad inserts 拿鸭的; bad deletes 非常多只鸭 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六位记者送给了你非常多只鸭。<br>Bad: 这六位记者送给了拿鸭的你。 |
| multiple edits: bad inserts 收; 乐家 -> 机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈的音乐家把手打断了。<br>Bad: 周大妈的收音机把手打断了。 |
| multiple edits: bad inserts 日; bad deletes 者 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她的记者演奏过奏鸣曲了。<br>Bad: 她的日记演奏过奏鸣曲了。 |
| multiple edits: 他一头 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个儿子借给了他一头大象。<br>Bad: 这个儿子借给了拿大象的他。 |
| multiple edits: 他五头 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个小孩递给了他五头大象。<br>Bad: 那个小孩递给了拿大象的他。 |
| multiple edits: 他八个 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你的上级送给了他八个舞者。<br>Bad: 你的上级送给了拿舞者的他。 |
| multiple edits: 他几个 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位打工人借给了他几个钢琴家。<br>Bad: 那位打工人借给了拿钢琴家的他。 |
| multiple edits: 他几条 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的弟弟卖给了他几条小狗。<br>Bad: 他们的弟弟卖给了拿小狗的他。 |
| multiple edits: 他四位 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个老板借给了他四位空姐。<br>Bad: 这个老板借给了拿空姐的他。 |
| multiple edits: 他好几十个 -> 拿; bad inserts 的他 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那三位演员借给了他好几十个老板。<br>Bad: 那三位演员借给了拿老板的他。 |
| multiple edits: 你一头 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八个司机送给了你一头大象。<br>Bad: 这八个司机送给了拿大象的你。 |
| multiple edits: 你三个 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位吉他手送给了你三个下属。<br>Bad: 这位吉他手送给了拿下属的你。 |
| multiple edits: 你四位 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位母亲送给了你四位学生。<br>Bad: 这位母亲送给了拿学生的你。 |
| multiple edits: 你好几个 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的哥哥借给了你好几个服务员。<br>Bad: 他们的哥哥借给了拿服务员的你。 |
| multiple edits: 你好几十个 -> 拿; bad inserts 的你 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生的老板卖给了你好几十个上级。<br>Bad: 张先生的老板卖给了拿上级的你。 |
| multiple edits: 她们几个 -> 拿; bad inserts 的她们 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八个消费者递给了她们几个音乐家。<br>Bad: 这八个消费者递给了拿音乐家的她们。 |
| multiple edits: 她八只 -> 拿; bad inserts 的她 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四的妹妹寄给了她八只老虎。<br>Bad: 李四的妹妹寄给了拿老虎的她。 |
| multiple edits: 她几头 -> 拿; bad inserts 的她 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三个演员买给了她几头大象。<br>Bad: 这三个演员买给了拿大象的她。 |
| multiple edits: 小王八个 -> 拿; bad inserts 的小王 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这四位下属寄给了小王八个消费者。<br>Bad: 这四位下属寄给了拿消费者的小王。 |
| multiple edits: 我一头 -> 拿; bad inserts 的我 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个朋友递给了我一头大象。<br>Bad: 这个朋友递给了拿大象的我。 |
| multiple edits: 我三个 -> 拿; bad inserts 的我 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位老板买给了我三个司机。<br>Bad: 这位老板买给了拿司机的我。 |
| multiple edits: 我五个 -> 拿; bad inserts 的我 | intransitive_double_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这四位舞者买给了我五个打工人。<br>Bad: 这四位舞者买给了拿打工人的我。 |
| multiple edits: 我五头 -> 拿; bad inserts 的我 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太的同事递给了我五头大象。<br>Bad: 何太太的同事递给了拿大象的我。 |
| multiple edits: 我们七位 -> 拿; bad inserts 的我们 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外五位下属寄给了我们七位演奏员。<br>Bad: 另外五位下属寄给了拿演奏员的我们。 |
| multiple edits: 我们十几个 -> 拿; bad inserts 的我们 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的上级寄给了我们十几个消费者。<br>Bad: 他的上级寄给了拿消费者的我们。 |
| multiple edits: 我们好几位 -> 拿; bad inserts 的我们 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那七位母亲买给了我们好几位演奏员。<br>Bad: 那七位母亲买给了拿演奏员的我们。 |
| multiple edits: 我十几位 -> 拿; bad inserts 的我 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈的下属寄给了我十几位服务员。<br>Bad: 郑大妈的下属寄给了拿服务员的我。 |
| multiple edits: 王五好几百位 -> 拿; bad inserts 的王五 | intransitive_double_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位下属卖给了王五好几百位消费者。<br>Bad: 这位下属卖给了拿消费者的王五。 |
| 上级 → 教材 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我的上级预习了教材。<br>Bad: 我的教材预习了教材。 |
| 上级 → 椅子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生的上级包扎了脚。<br>Bad: 刘先生的椅子包扎了脚。 |
| 上级 → 糖果 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 上级盖过被子了。<br>Bad: 糖果盖过被子了。 |
| 上级 → 裤子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 上级领养了小猫。<br>Bad: 裤子领养了小猫。 |
| 上级 → 视频 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的上级喝了冰红茶。<br>Bad: 李太太的视频喝了冰红茶。 |
| 下属 → 作业 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 下属把糖果吃了。<br>Bad: 作业把糖果吃了。 |
| 下属 → 咖啡 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷的下属领养了小猫。<br>Bad: 胡大爷的咖啡领养了小猫。 |
| 下属 → 手账 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 下属驾驶了货车。<br>Bad: 手账驾驶了货车。 |
| 下属 → 椅子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 下属煮过鸭了。<br>Bad: 椅子煮过鸭了。 |
| 下属 → 被子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明的下属把鱼煮了。<br>Bad: 小明的被子把鱼煮了。 |
| 他们 → 作业 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九瓶冰红茶被他们喝了。<br>Bad: 这九瓶冰红茶被作业喝了。 |
| 他们 → 双簧 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那一瓶红酒被他们喝了。<br>Bad: 那一瓶红酒被双簧喝了。 |
| 他们 → 可乐 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外八个玻璃珠被他们盖了。<br>Bad: 另外八个玻璃珠被可乐盖了。 |
| 他们 → 咖啡 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那条小狗被他们领养了。<br>Bad: 那条小狗被咖啡领养了。 |
| 他们 → 戏曲 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这桶矿泉水被他们喝了。<br>Bad: 这桶矿泉水被戏曲喝了。 |
| 他们 → 橙汁 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个头被他们打断了。<br>Bad: 那个头被橙汁打断了。 |
| 他们 → 漫画 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这十只手被他们检查了。<br>Bad: 这十只手被漫画检查了。 |
| 他们 → 红酒 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那只鸡被他们炖了。<br>Bad: 那只鸡被红酒炖了。 |
| 他们 → 面包 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那部记录片被他们观看了。<br>Bad: 那部记录片被面包观看了。 |
| 他们 → 香蕉 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七本书被他们看了。<br>Bad: 这七本书被香蕉看了。 |
| 何太太 → 方便面 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十部电影被何太太制作了。<br>Bad: 另外十部电影被方便面制作了。 |
| 何太太 → 电冰箱 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这头牛被何太太屠宰了。<br>Bad: 这头牛被电冰箱屠宰了。 |
| 你 → 书 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外两桶方便面被你吃了。<br>Bad: 另外两桶方便面被书吃了。 |
| 你们 → 作业 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这块糖果被你们吃了。<br>Bad: 这块糖果被作业吃了。 |
| 你们 → 日记 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个杯子被你们喝了。<br>Bad: 那个杯子被日记喝了。 |
| 你们 → 桌子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这一头大象被你们吃了。<br>Bad: 这一头大象被桌子吃了。 |
| 你们 → 沙漠 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那条鱼被你们烧了。<br>Bad: 那条鱼被沙漠烧了。 |
| 你们 → 牛奶 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那十只手套被你们清洗了。<br>Bad: 那十只手套被牛奶清洗了。 |
| 你们 → 白酒 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外十个开瓶器被你们卖了。<br>Bad: 另外十个开瓶器被白酒卖了。 |
| 你们 → 红酒 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这瓶啤酒被你们买了。<br>Bad: 这瓶啤酒被红酒买了。 |
| 你们 → 美声 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这张桌子被你们搬了。<br>Bad: 这张桌子被美声搬了。 |
| 你们 → 被子 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这张桌子被你们看了。<br>Bad: 这张桌子被被子看了。 |
| 你们 → 面包 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那部电影被你们观看了。<br>Bad: 那部电影被面包观看了。 |
| 停下 → 原谅 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们停下了。<br>Bad: 他们原谅了。 |
| 停下 → 取缔 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们停下了。<br>Bad: 她们取缔了。 |
| 停下 → 夸奖 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷的母亲停下了。<br>Bad: 胡大爷的母亲夸奖了。 |
| 停下 → 完成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太的学生停下了。<br>Bad: 何太太的学生完成了。 |
| 停下 → 找到 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的领导停下了。<br>Bad: 她们的领导找到了。 |
| 停下 → 拥护 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四的下属停下了。<br>Bad: 李四的下属拥护了。 |
| 停下 → 排挤 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位工人停下了。<br>Bad: 那位工人排挤了。 |
| 停下 → 教育 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太停下了。<br>Bad: 何太太教育了。 |
| 停下 → 爱戴 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的姐妹停下了。<br>Bad: 他们的姐妹爱戴了。 |
| 停下 → 相信 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外十位记者停下了。<br>Bad: 另外十位记者相信了。 |
| 停下 → 称赞 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明停下了。<br>Bad: 小明称赞了。 |
| 停下 → 维护 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生的同事停下了。<br>Bad: 张先生的同事维护了。 |
| 停下 → 追捧 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们停下了。<br>Bad: 我们追捧了。 |
| 健身 → 取代 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷健身了。<br>Bad: 赵大爷取代了。 |
| 健身 → 嘉奖 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们健身了。<br>Bad: 她们嘉奖了。 |
| 健身 → 控制 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个吉他手健身了。<br>Bad: 这个吉他手控制了。 |
| 健身 → 表扬 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我健身了。<br>Bad: 我表扬了。 |
| 偷听 → 控制 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太偷听了。<br>Bad: 何太太控制了。 |
| 偷听 → 支持 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九个司机偷听了。<br>Bad: 那九个司机支持了。 |
| 偷听 → 清蒸 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈的朋友偷听了。<br>Bad: 周大妈的朋友清蒸了。 |
| 偷听 → 赞成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘偷听了。<br>Bad: 王大娘赞成了。 |
| 偷听 → 驾驶 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那四位老师偷听了。<br>Bad: 那四位老师驾驶了。 |
| 儿 → 袜 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 儿子捕捉了蛇。<br>Bad: 袜子捕捉了蛇。 |
| 儿子 → 作业 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 儿子把肚子包扎了。<br>Bad: 作业把肚子包扎了。 |
| 儿子 → 双簧 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太的儿子喝了葡萄汁。<br>Bad: 何太太的双簧喝了葡萄汁。 |
| 儿子 → 可乐 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 儿子检查过肚子了。<br>Bad: 可乐检查过肚子了。 |
| 兄弟 → 白酒 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的兄弟演奏了歌曲。<br>Bad: 你们的白酒演奏了歌曲。 |
| 兄弟 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我的兄弟麻醉过大象了。<br>Bad: 我的袜子麻醉过大象了。 |
| 入睡 → 厌恶 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的母亲入睡了。<br>Bad: 我的母亲厌恶了。 |
| 入睡 → 完成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们入睡了。<br>Bad: 她们完成了。 |
| 入睡 → 提醒 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位吉他手入睡了。<br>Bad: 这位吉他手提醒了。 |
| 入睡 → 表扬 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们的上级入睡了。<br>Bad: 我们的上级表扬了。 |
| 入睡 → 观看 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们入睡了。<br>Bad: 你们观看了。 |
| 冯大哥 → 方便面 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那五个观点被冯大哥抨击了。<br>Bad: 那五个观点被方便面抨击了。 |
| 冯大哥 → 电视剧 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那本手账被冯大哥制作了。<br>Bad: 那本手账被电视剧制作了。 |
| 出发 → 呵斥 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶的女儿出发了。<br>Bad: 张婶的女儿呵斥了。 |
| 出发 → 埋怨 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她的上级出发了。<br>Bad: 她的上级埋怨了。 |
| 出发 → 建立 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位学生出发了。<br>Bad: 这位学生建立了。 |
| 出发 → 打断 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四出发了。<br>Bad: 李四打断了。 |
| 出发 → 批评 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位服务员出发了。<br>Bad: 那位服务员批评了。 |
| 出发 → 相信 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她的员工出发了。<br>Bad: 她的员工相信了。 |
| 刘先生 → 照相馆 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那把椅子被刘先生搬了。<br>Bad: 那把椅子被照相馆搬了。 |
| 刘先生 → 电视机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这把椅子被刘先生搬了。<br>Bad: 这把椅子被电视机搬了。 |
| 创作 → 凝固 | agent_causative | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太的员工几乎创作了漫画。<br>Bad: 吴太太的员工几乎凝固了漫画。 |
| 创作 → 蒸发 | agent_causative | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人差点儿创作了漫画。<br>Bad: 张夫人差点儿蒸发了漫画。 |
| 创作 → 融化 | agent_causative | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈差点儿创作了小说。<br>Bad: 郑大妈差点儿融化了小说。 |
| 制作 → 凝固 | agent_causative | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他制作了电影。<br>Bad: 他凝固了电影。 |
| 包扎 → 凝固 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八位舞者包扎了肚子。<br>Bad: 这八位舞者凝固了肚子。 |
| 包扎 → 出现 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位母亲包扎了腿。<br>Bad: 这位母亲出现了腿。 |
| 去 → 开 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐去了。<br>Bad: 陈大姐开了。 |
| 去 → 炖 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外两个下属去了。<br>Bad: 另外两个下属炖了。 |
| 司机 → 作业 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 司机把咖啡喝了。<br>Bad: 作业把咖啡喝了。 |
| 司机 → 手套 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶的司机把飞机开了。<br>Bad: 张婶的手套把飞机开了。 |
| 司机 → 教材 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生的司机捕捉了鱼。<br>Bad: 王先生的教材捕捉了鱼。 |
| 司机 → 椅子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶的司机把牛屠宰了。<br>Bad: 张婶的椅子把牛屠宰了。 |
| 司机 → 橘子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 司机唱了戏曲。<br>Bad: 橘子唱了戏曲。 |
| 司机 → 电影 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 司机把手账制作了。<br>Bad: 电影把手账制作了。 |
| 司机 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的司机把杯子清洗了。<br>Bad: 你们的袜子把杯子清洗了。 |
| 司机 → 裙子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 司机吹了双簧。<br>Bad: 裙子吹了双簧。 |
| 司机 → 钢琴 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 司机把漫画创作了。<br>Bad: 钢琴把漫画创作了。 |
| 叹息 → 安慰 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位同事叹息了。<br>Bad: 这位同事安慰了。 |
| 叹息 → 打劫 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他叹息了。<br>Bad: 他打劫了。 |
| 叹息 → 批评 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的学生叹息了。<br>Bad: 我的学生批评了。 |
| 叹息 → 演奏 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五位老师叹息了。<br>Bad: 这五位老师演奏了。 |
| 叹息 → 麻醉 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们叹息了。<br>Bad: 他们麻醉了。 |
| 吉他手 → 收音机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吉他手盖过被子了。<br>Bad: 收音机盖过被子了。 |
| 吉他手 → 电冰箱 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的吉他手创作了小说。<br>Bad: 她们的电冰箱创作了小说。 |
| 同事 → 桌子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的同事屠宰了牛。<br>Bad: 李太太的桌子屠宰了牛。 |
| 同事 → 电影 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷的同事演奏过协奏曲了。<br>Bad: 赵大爷的电影演奏过协奏曲了。 |
| 同事 → 裤子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生的同事拍摄过电影了。<br>Bad: 张先生的裤子拍摄过电影了。 |
| 听课 → 伤害 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三位演奏员听课了。<br>Bad: 这三位演奏员伤害了。 |
| 听课 → 找到 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个司机听课了。<br>Bad: 那个司机找到了。 |
| 听课 → 支持 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们听课了。<br>Bad: 我们支持了。 |
| 听课 → 检查 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的学生听课了。<br>Bad: 你的学生检查了。 |
| 听课 → 登上 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个消费者听课了。<br>Bad: 那个消费者登上了。 |
| 听课 → 观看 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们听课了。<br>Bad: 她们观看了。 |
| 听课 → 追捧 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太听课了。<br>Bad: 何太太追捧了。 |
| 启程 → 包扎 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们启程了。<br>Bad: 她们包扎了。 |
| 启程 → 反驳 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人启程了。<br>Bad: 张夫人反驳了。 |
| 启程 → 打劫 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这四个小孩启程了。<br>Bad: 这四个小孩打劫了。 |
| 启程 → 控制 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五个姐姐启程了。<br>Bad: 这五个姐姐控制了。 |
| 启程 → 推崇 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个老板启程了。<br>Bad: 这个老板推崇了。 |
| 启程 → 支持 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥启程了。<br>Bad: 冯大哥支持了。 |
| 启程 → 表扬 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶启程了。<br>Bad: 张婶表扬了。 |
| 启程 → 跨越 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘的下属启程了。<br>Bad: 王大娘的下属跨越了。 |
| 启程 → 重建 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥启程了。<br>Bad: 杨大哥重建了。 |
| 吴太太 → 小提琴 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外四部漫画被吴太太观看了。<br>Bad: 另外四部漫画被小提琴观看了。 |
| 吴太太 → 收音机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五部动作片被吴太太拍摄了。<br>Bad: 这五部动作片被收音机拍摄了。 |
| 吴太太 → 玻璃珠 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位学生被吴太太称赞了。<br>Bad: 这位学生被玻璃珠称赞了。 |
| 吴太太 → 电视机 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那桶啤酒被吴太太喝了。<br>Bad: 那桶啤酒被电视机喝了。 |
| 员工 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们的员工看了教材。<br>Bad: 我们的杯子看了教材。 |
| 员工 → 电影 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷的员工驾驶过飞机了。<br>Bad: 胡大爷的电影驾驶过飞机了。 |
| 员工 → 香蕉 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的员工制作了动画片。<br>Bad: 你们的香蕉制作了动画片。 |
| 周大妈 → 充电器 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那六张桌子被周大妈搬了。<br>Bad: 那六张桌子被充电器搬了。 |
| 周大妈 → 开瓶器 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外三部电影被周大妈观看了。<br>Bad: 另外三部电影被开瓶器观看了。 |
| 周大妈 → 录像带 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那块糖被周大妈吃了。<br>Bad: 那块糖被录像带吃了。 |
| 周大妈 → 收音机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这只鸭被周大妈爆炒了。<br>Bad: 这只鸭被收音机爆炒了。 |
| 周大妈 → 方便面 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个想法被周大妈维护了。<br>Bad: 这个想法被方便面维护了。 |
| 周大妈 → 饮料瓶 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九张桌子被周大妈搬了。<br>Bad: 那九张桌子被饮料瓶搬了。 |
| 呼吸 → 制作 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐呼吸了。<br>Bad: 徐小姐制作了。 |
| 呼吸 → 厌恶 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你呼吸了。<br>Bad: 你厌恶了。 |
| 呼吸 → 批评 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他的下属呼吸了。<br>Bad: 他的下属批评了。 |
| 呼吸 → 欺骗 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五位空姐呼吸了。<br>Bad: 这五位空姐欺骗了。 |
| 呼吸 → 鼓励 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他呼吸了。<br>Bad: 他鼓励了。 |
| 品茶 → 原谅 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我品茶了。<br>Bad: 我原谅了。 |
| 品茶 → 回到 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位舞者品茶了。<br>Bad: 那位舞者回到了。 |
| 品茶 → 夸奖 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐品茶了。<br>Bad: 王小姐夸奖了。 |
| 品茶 → 屠宰 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她品茶了。<br>Bad: 她屠宰了。 |
| 品茶 → 打劫 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们的领导品茶了。<br>Bad: 我们的领导打劫了。 |
| 品茶 → 打断 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位服务员品茶了。<br>Bad: 那位服务员打断了。 |
| 品茶 → 爱护 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那三位老师品茶了。<br>Bad: 那三位老师爱护了。 |
| 品茶 → 表扬 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥的兄弟品茶了。<br>Bad: 冯大哥的兄弟表扬了。 |
| 品茶 → 重建 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们品茶了。<br>Bad: 我们重建了。 |
| 哥哥 → 衣服 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐的哥哥把笛子吹了。<br>Bad: 王小姐的衣服把笛子吹了。 |
| 哥哥 → 馒头 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你的哥哥把小狗领养了。<br>Bad: 你的馒头把小狗领养了。 |
| 哭 → 摆 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个打工人哭了。<br>Bad: 那个打工人摆了。 |
| 唱歌 → 创作 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的儿子唱歌了。<br>Bad: 她的儿子创作了。 |
| 唱歌 → 嘉奖 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外八位员工唱歌了。<br>Bad: 另外八位员工嘉奖了。 |
| 唱歌 → 完成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个工人唱歌了。<br>Bad: 这个工人完成了。 |
| 唱歌 → 批判 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们唱歌了。<br>Bad: 她们批判了。 |
| 唱歌 → 推崇 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位消费者唱歌了。<br>Bad: 这位消费者推崇了。 |
| 唱歌 → 赞成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太唱歌了。<br>Bad: 何太太赞成了。 |
| 坐下 → 反驳 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐坐下了。<br>Bad: 王小姐反驳了。 |
| 坐下 → 呵斥 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太坐下了。<br>Bad: 吴太太呵斥了。 |
| 坐下 → 喜欢 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们坐下了。<br>Bad: 我们喜欢了。 |
| 坐下 → 批评 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五坐下了。<br>Bad: 王五批评了。 |
| 坐下 → 抨击 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥坐下了。<br>Bad: 冯大哥抨击了。 |
| 坐下 → 相信 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外三个朋友坐下了。<br>Bad: 另外三个朋友相信了。 |
| 坐下 → 预习 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那五个吉他手坐下了。<br>Bad: 那五个吉他手预习了。 |
| 女儿 → 小说 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的女儿观看了电视剧。<br>Bad: 她的小说观看了电视剧。 |
| 女儿 → 电影 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的女儿清蒸了鸭。<br>Bad: 他们的电影清蒸了鸭。 |
| 奴隶 → 桌子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 奴隶把小提琴拉了。<br>Bad: 桌子把小提琴拉了。 |
| 奴隶 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们的奴隶预习过教材了。<br>Bad: 她们的袜子预习过教材了。 |
| 妈妈 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐的妈妈把鱼炖了。<br>Bad: 王小姐的杯子把鱼炖了。 |
| 妈妈 → 花卷 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的妈妈打断过腿了。<br>Bad: 她们的花卷打断过腿了。 |
| 妹妹 → 教材 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 妹妹把小说创作了。<br>Bad: 教材把小说创作了。 |
| 姐妹 → 教材 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 姐妹把脚打断了。<br>Bad: 教材把脚打断了。 |
| 姐妹 → 电影 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 姐妹炖过鱼了。<br>Bad: 电影炖过鱼了。 |
| 姐姐 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 姐姐把方便面吃了。<br>Bad: 袜子把方便面吃了。 |
| 姐姐 → 裤子 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 姐姐把小狗领养了。<br>Bad: 裤子把小狗领养了。 |
| 学生 → 衣服 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的学生煮了鸡。<br>Bad: 我的衣服煮了鸡。 |
| 宋女士 → 开瓶器 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那八串香蕉被宋女士买了。<br>Bad: 那八串香蕉被开瓶器买了。 |
| 宋女士 → 热水器 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那头牛被宋女士屠宰了。<br>Bad: 那头牛被热水器屠宰了。 |
| 宋女士 → 电视机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这张桌子被宋女士清洗了。<br>Bad: 这张桌子被电视机清洗了。 |
| 宋女士 → 葡萄汁 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外两把椅子被宋女士搬了。<br>Bad: 另外两把椅子被葡萄汁搬了。 |
| 宋女士 → 饮料瓶 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这头牛被宋女士屠宰了。<br>Bad: 这头牛被饮料瓶屠宰了。 |
| 小孩 → 教材 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘的小孩包扎过手了。<br>Bad: 王大娘的教材包扎过手了。 |
| 小孩 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生的小孩唱了歌。<br>Bad: 刘先生的杯子唱了歌。 |
| 小明 → 漫画 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个儿子被小明埋怨了。<br>Bad: 这个儿子被漫画埋怨了。 |
| 小明 → 被子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外两头牛被小明麻醉了。<br>Bad: 另外两头牛被被子麻醉了。 |
| 小明 → 裤子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外七个饮料瓶被小明买了。<br>Bad: 另外七个饮料瓶被裤子买了。 |
| 小王 → 可乐 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那头牛被小王屠宰了。<br>Bad: 那头牛被可乐屠宰了。 |
| 小王 → 桌子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这两条腿被小王包扎了。<br>Bad: 这两条腿被桌子包扎了。 |
| 小王 → 裙子 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一块糖果被小王买了。<br>Bad: 另外一块糖果被裙子买了。 |
| 小王 → 裤子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那九个饮料瓶被小王吃了。<br>Bad: 那九个饮料瓶被裤子吃了。 |
| 屠宰 → 存在 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个服务员屠宰了牛。<br>Bad: 那个服务员存在了牛。 |
| 工人 → 日记 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三的工人吹了双簧。<br>Bad: 张三的日记吹了双簧。 |
| 工人 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们的工人把鱼捕捉了。<br>Bad: 我们的杯子把鱼捕捉了。 |
| 工人 → 橙汁 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨的工人拍摄过电影了。<br>Bad: 王姨的橙汁拍摄过电影了。 |
| 工人 → 蛋糕 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 工人打断了鼻子。<br>Bad: 蛋糕打断了鼻子。 |
| 工人 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你的工人把小说创作了。<br>Bad: 你的袜子把小说创作了。 |
| 弟弟 → 啤酒 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生的弟弟跨越过沙漠了。<br>Bad: 王先生的啤酒跨越过沙漠了。 |
| 张三 → 可乐 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外四瓶可乐被张三买了。<br>Bad: 另外四瓶可乐被可乐买了。 |
| 张三 → 小说 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这八部小说被张三创作了。<br>Bad: 这八部小说被小说创作了。 |
| 张三 → 教材 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九张桌子被张三盖了。<br>Bad: 这九张桌子被教材盖了。 |
| 张三 → 红酒 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外六串香蕉被张三买了。<br>Bad: 另外六串香蕉被红酒买了。 |
| 张三 → 袜子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这四条裙子被张三预习了。<br>Bad: 这四条裙子被袜子预习了。 |
| 张先生 → 玻璃珠 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一部教材被张先生看了。<br>Bad: 这一部教材被玻璃珠看了。 |
| 张先生 → 电视机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那只脚被张先生检查了。<br>Bad: 那只脚被电视机检查了。 |
| 张夫人 → 巧克力 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这六桶方便面被张夫人吃了。<br>Bad: 这六桶方便面被巧克力吃了。 |
| 张婶 → 椅子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这九条腿被张婶打断了。<br>Bad: 这九条腿被椅子打断了。 |
| 张婶 → 白酒 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外两个花卷被张婶吃了。<br>Bad: 另外两个花卷被白酒吃了。 |
| 我 → 书 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这串香蕉被我买了。<br>Bad: 这串香蕉被书买了。 |
| 我 → 歌 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个眼睛被我包扎了。<br>Bad: 这个眼睛被歌包扎了。 |
| 我们 → 作业 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那条裤子被我们喝了。<br>Bad: 那条裤子被作业喝了。 |
| 我们 → 小说 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这只老虎被我们麻醉了。<br>Bad: 这只老虎被小说麻醉了。 |
| 我们 → 戏曲 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那把椅子被我们清洗了。<br>Bad: 那把椅子被戏曲清洗了。 |
| 我们 → 椅子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那头大象被我们麻醉了。<br>Bad: 那头大象被椅子麻醉了。 |
| 我们 → 电影 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六杯牛奶被我们喝了。<br>Bad: 这六杯牛奶被电影喝了。 |
| 我们 → 花卷 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那一桶啤酒被我们喝了。<br>Bad: 那一桶啤酒被花卷喝了。 |
| 我们 → 衣服 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这条鱼被我们吃了。<br>Bad: 这条鱼被衣服吃了。 |
| 我们 → 裙子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那一条腿被我们治疗了。<br>Bad: 那一条腿被裙子治疗了。 |
| 打工人 → 巧克力 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 打工人预习了教材。<br>Bad: 巧克力预习了教材。 |
| 打工人 → 收音机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四的打工人捕捉了鸡。<br>Bad: 李四的收音机捕捉了鸡。 |
| 打工人 → 热水器 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 打工人弹了玻璃珠。<br>Bad: 热水器弹了玻璃珠。 |
| 打工人 → 电冰箱 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 打工人创作过漫画了。<br>Bad: 电冰箱创作过漫画了。 |
| 打工人 → 电视机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生的打工人把老虎捕捉了。<br>Bad: 刘先生的电视机把老虎捕捉了。 |
| 打断 → 融化 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐打断了鼻子。<br>Bad: 王小姐融化了鼻子。 |
| 打架 → 反感 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太的妈妈打架了。<br>Bad: 吴太太的妈妈反感了。 |
| 打架 → 建立 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三打架了。<br>Bad: 张三建立了。 |
| 打架 → 批判 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这四位吉他手打架了。<br>Bad: 这四位吉他手批判了。 |
| 打架 → 找到 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨打架了。<br>Bad: 王姨找到了。 |
| 打架 → 捕捉 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三的母亲打架了。<br>Bad: 张三的母亲捕捉了。 |
| 打架 → 支持 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的下属打架了。<br>Bad: 你的下属支持了。 |
| 打架 → 演奏 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们打架了。<br>Bad: 他们演奏了。 |
| 打架 → 鼓励 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈打架了。<br>Bad: 周大妈鼓励了。 |
| 拍摄 → 消失 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三几乎拍摄了电影。<br>Bad: 张三几乎消失了电影。 |
| 朋友 → 咖啡 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的朋友把被子盖了。<br>Bad: 她的咖啡把被子盖了。 |
| 朋友 → 手套 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 朋友看了书。<br>Bad: 手套看了书。 |
| 朋友 → 教材 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 朋友拍摄过电影了。<br>Bad: 教材拍摄过电影了。 |
| 朋友 → 池塘 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的朋友烧了鱼。<br>Bad: 你们的池塘烧了鱼。 |
| 服务员 → 录像带 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈的服务员清洗了杯子。<br>Bad: 郑大妈的录像带清洗了杯子。 |
| 服务员 → 电冰箱 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的服务员捕捉过大象了。<br>Bad: 李太太的电冰箱捕捉过大象了。 |
| 服务员 → 电影院 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 服务员清洗了杯子。<br>Bad: 电影院清洗了杯子。 |
| 李先生 → 开瓶器 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这瓶可乐被李先生买了。<br>Bad: 这瓶可乐被开瓶器买了。 |
| 李四 → 啤酒 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这片面包被李四吃了。<br>Bad: 这片面包被啤酒吃了。 |
| 李四 → 桌子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那八张桌子被李四搬了。<br>Bad: 那八张桌子被桌子搬了。 |
| 李四 → 糖果 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那块糖被李四吃了。<br>Bad: 那块糖被糖果吃了。 |
| 李四 → 袜子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外四片面包被李四吃了。<br>Bad: 另外四片面包被袜子吃了。 |
| 李四 → 裙子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那串香蕉被李四吃了。<br>Bad: 那串香蕉被裙子吃了。 |
| 李四 → 裤子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这只鸡被李四麻醉了。<br>Bad: 这只鸡被裤子麻醉了。 |
| 李太太 → 录像带 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这三把椅子被李太太吃了。<br>Bad: 这三把椅子被录像带吃了。 |
| 李太太 → 方便面 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这八个玻璃珠被李太太喝了。<br>Bad: 这八个玻璃珠被方便面喝了。 |
| 来 → 吃 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生来了。<br>Bad: 王先生吃了。 |
| 来 → 开 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生的妹妹来了。<br>Bad: 王先生的妹妹开了。 |
| 母亲 → 作业 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 母亲把头包扎了。<br>Bad: 作业把头包扎了。 |
| 消费者 → 华尔兹 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 消费者把飞机驾驶了。<br>Bad: 华尔兹把飞机驾驶了。 |
| 消费者 → 收音机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 消费者把华尔兹演奏了。<br>Bad: 收音机把华尔兹演奏了。 |
| 消防员 → 收音机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 消防员制作了手账。<br>Bad: 收音机制作了手账。 |
| 游泳 → 拥护 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们游泳了。<br>Bad: 我们拥护了。 |
| 游泳 → 清洗 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五的朋友游泳了。<br>Bad: 王五的朋友清洗了。 |
| 游泳 → 演奏 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那个吉他手游泳了。<br>Bad: 那个吉他手演奏了。 |
| 溜走 → 厌恶 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那六位学生溜走了。<br>Bad: 那六位学生厌恶了。 |
| 溜走 → 取代 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们溜走了。<br>Bad: 你们取代了。 |
| 溜走 → 嫌弃 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的员工溜走了。<br>Bad: 你们的员工嫌弃了。 |
| 溜走 → 拍摄 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五的母亲溜走了。<br>Bad: 王五的母亲拍摄了。 |
| 溜走 → 追捧 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们溜走了。<br>Bad: 她们追捧了。 |
| 演员 → 啤酒 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明的演员唱过美声了。<br>Bad: 小明的啤酒唱过美声了。 |
| 演员 → 小说 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 演员唱过戏曲了。<br>Bad: 小说唱过戏曲了。 |
| 演员 → 手账 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 演员把手打断了。<br>Bad: 手账把手打断了。 |
| 演员 → 椅子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈的演员炖了鸡。<br>Bad: 郑大妈的椅子炖了鸡。 |
| 演员 → 糖果 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 演员爆炒了鸡。<br>Bad: 糖果爆炒了鸡。 |
| 演员 → 裤子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨的演员盖过被子了。<br>Bad: 王姨的裤子盖过被子了。 |
| 演奏 → 变化 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士差点儿演奏了歌曲。<br>Bad: 宋女士差点儿变化了歌曲。 |
| 演奏员 → 大提琴 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 演奏员预习过教材了。<br>Bad: 大提琴预习过教材了。 |
| 演奏员 → 热水器 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 演奏员把书看了。<br>Bad: 热水器把书看了。 |
| 爆炒 → 出现 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太爆炒了鸭。<br>Bad: 李太太出现了鸭。 |
| 爆炒 → 变化 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐险些爆炒了鱼。<br>Bad: 陈大姐险些变化了鱼。 |
| 爆炒 → 存在 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们爆炒了鸭。<br>Bad: 你们存在了鸭。 |
| 爬行 → 原谅 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们爬行了。<br>Bad: 她们原谅了。 |
| 父亲 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太的父亲把华尔兹演奏了。<br>Bad: 吴太太的杯子把华尔兹演奏了。 |
| 爸爸 → 红茶 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的爸爸跨越了沙漠。<br>Bad: 她的红茶跨越了沙漠。 |
| 王五 → 手套 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这只老虎被王五捕捉了。<br>Bad: 这只老虎被手套捕捉了。 |
| 王五 → 袜子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外两桶方便面被王五吃了。<br>Bad: 另外两桶方便面被袜子吃了。 |
| 王先生 → 巧克力 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那头大象被王先生麻醉了。<br>Bad: 那头大象被巧克力麻醉了。 |
| 王先生 → 玻璃珠 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个演员被王先生喜欢了。<br>Bad: 那个演员被玻璃珠喜欢了。 |
| 王大娘 → 热水器 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这八个观点被王大娘维护了。<br>Bad: 这八个观点被热水器维护了。 |
| 王大娘 → 电影院 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这桶方便面被王大娘买了。<br>Bad: 这桶方便面被电影院买了。 |
| 王姨 → 教材 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九只鸭被王姨炖了。<br>Bad: 那九只鸭被教材炖了。 |
| 王姨 → 杯子 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那八把椅子被王姨预习了。<br>Bad: 那八把椅子被杯子预习了。 |
| 王姨 → 漫画 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五部手账被王姨观看了。<br>Bad: 这五部手账被漫画观看了。 |
| 王姨 → 衣服 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个理念被王姨辩护了。<br>Bad: 这个理念被衣服辩护了。 |
| 王小姐 → 开瓶器 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这杯橙汁被王小姐喝了。<br>Bad: 这杯橙汁被开瓶器喝了。 |
| 王小姐 → 电视机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这三头大象被王小姐捕捉了。<br>Bad: 这三头大象被电视机捕捉了。 |
| 玩耍 → 夸奖 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生的弟弟玩耍了。<br>Bad: 刘先生的弟弟夸奖了。 |
| 玩耍 → 尊重 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个姐姐玩耍了。<br>Bad: 那个姐姐尊重了。 |
| 玩耍 → 爆炒 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨玩耍了。<br>Bad: 王姨爆炒了。 |
| 玩耍 → 维护 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外三个儿子玩耍了。<br>Bad: 另外三个儿子维护了。 |
| 玩耍 → 责备 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位演员玩耍了。<br>Bad: 那位演员责备了。 |
| 玩耍 → 赞成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们玩耍了。<br>Bad: 我们赞成了。 |
| 看戏 → 拍摄 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥看戏了。<br>Bad: 冯大哥拍摄了。 |
| 睡觉 → 嘉奖 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈睡觉了。<br>Bad: 周大妈嘉奖了。 |
| 睡觉 → 拥护 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们睡觉了。<br>Bad: 我们拥护了。 |
| 睡觉 → 清蒸 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我的领导睡觉了。<br>Bad: 我的领导清蒸了。 |
| 睡觉 → 相信 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四睡觉了。<br>Bad: 李四相信了。 |
| 睡觉 → 约束 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的老板睡觉了。<br>Bad: 你的老板约束了。 |
| 睡觉 → 领养 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘睡觉了。<br>Bad: 王大娘领养了。 |
| 空姐 → 红茶 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王的空姐把鸡炖了。<br>Bad: 小王的红茶把鸡炖了。 |
| 站立 → 厌恶 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈站立了。<br>Bad: 周大妈厌恶了。 |
| 站立 → 原谅 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们站立了。<br>Bad: 我们原谅了。 |
| 罪犯 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明的罪犯把鸭捕捉了。<br>Bad: 小明的杯子把鸭捕捉了。 |
| 罪犯 → 橙汁 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的罪犯检查过耳朵了。<br>Bad: 他们的橙汁检查过耳朵了。 |
| 罪犯 → 红酒 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人的罪犯清蒸过鱼了。<br>Bad: 张夫人的红酒清蒸过鱼了。 |
| 老师 → 手套 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的老师拉过大提琴了。<br>Bad: 你的手套拉过大提琴了。 |
| 老师 → 白酒 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人的老师把杯子清洗了。<br>Bad: 张夫人的白酒把杯子清洗了。 |
| 老师 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的老师喝过红茶了。<br>Bad: 他们的袜子喝过红茶了。 |
| 老师 → 被子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们的老师跨越过沙漠了。<br>Bad: 她们的被子跨越过沙漠了。 |
| 老板 → 电影 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的老板把鸡煮了。<br>Bad: 她们的电影把鸡煮了。 |
| 老板 → 笛子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的老板创作了小说。<br>Bad: 他们的笛子创作了小说。 |
| 老板 → 衣服 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 老板把教材预习了。<br>Bad: 衣服把教材预习了。 |
| 老板 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 老板拍摄过动画片了。<br>Bad: 袜子拍摄过动画片了。 |
| 胡大爷 → 玻璃珠 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五把椅子被胡大爷吃了。<br>Bad: 这五把椅子被玻璃珠吃了。 |
| 胡大爷 → 记录片 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那把椅子被胡大爷盖了。<br>Bad: 那把椅子被记录片盖了。 |
| 胡大爷 → 饮料瓶 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这只鸡被胡大爷吃了。<br>Bad: 这只鸡被饮料瓶吃了。 |
| 舞者 → 可乐 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 舞者爆炒过鸭了。<br>Bad: 可乐爆炒过鸭了。 |
| 舞者 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘的舞者把飞机驾驶了。<br>Bad: 王大娘的杯子把飞机驾驶了。 |
| 舞者 → 糖果 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 舞者创作过小说了。<br>Bad: 糖果创作过小说了。 |
| 舞者 → 衣服 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们的舞者创作了小说。<br>Bad: 她们的衣服创作了小说。 |
| 观看 → 凝固 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们差点儿观看了动画片。<br>Bad: 我们差点儿凝固了动画片。 |
| 观看 → 出现 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她几乎观看了记录片。<br>Bad: 她几乎出现了记录片。 |
| 观看 → 存在 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那两位打工人观看了电影。<br>Bad: 那两位打工人存在了电影。 |
| 观看 → 消失 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太差点儿观看了电影。<br>Bad: 何太太差点儿消失了电影。 |
| 警察 → 桌子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 警察打断过腿了。<br>Bad: 桌子打断过腿了。 |
| 警察 → 漫画 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人的警察预习过教材了。<br>Bad: 张夫人的漫画预习过教材了。 |
| 记者 → 京剧 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的记者打断了腿。<br>Bad: 你们的京剧打断了腿。 |
| 记者 → 小说 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 记者把教材预习了。<br>Bad: 小说把教材预习了。 |
| 记者 → 山洞 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生的记者跨越了海洋。<br>Bad: 李先生的山洞跨越了海洋。 |
| 记者 → 教材 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人的记者开了卡车。<br>Bad: 张夫人的教材开了卡车。 |
| 记者 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人的记者拍摄了记录片。<br>Bad: 张夫人的杯子拍摄了记录片。 |
| 记者 → 海洋 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士的记者把鸡炖了。<br>Bad: 宋女士的海洋把鸡炖了。 |
| 记者 → 糖果 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 记者炖了鸭。<br>Bad: 糖果炖了鸭。 |
| 走 → 吃 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈的同事走了。<br>Bad: 周大妈的同事吃了。 |
| 走 → 学 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位演员走了。<br>Bad: 这位演员学了。 |
| 走 → 拉 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他走了。<br>Bad: 他拉了。 |
| 走 → 炖 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士走了。<br>Bad: 宋女士炖了。 |
| 走 → 煮 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨走了。<br>Bad: 王姨煮了。 |
| 走路 → 夸奖 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的同事走路了。<br>Bad: 你们的同事夸奖了。 |
| 走路 → 拥护 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷走路了。<br>Bad: 赵大爷拥护了。 |
| 走路 → 捕捉 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九位学生走路了。<br>Bad: 那九位学生捕捉了。 |
| 走路 → 检查 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥走路了。<br>Bad: 冯大哥检查了。 |
| 走路 → 清蒸 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五个小孩走路了。<br>Bad: 这五个小孩清蒸了。 |
| 走路 → 赞成 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九位音乐家走路了。<br>Bad: 这九位音乐家赞成了。 |
| 走路 → 重建 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈的妈妈走路了。<br>Bad: 周大妈的妈妈重建了。 |
| 赵大爷 → 巧克力 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那一本教材被赵大爷预习了。<br>Bad: 那一本教材被巧克力预习了。 |
| 赵大爷 → 玻璃珠 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个眼睛被赵大爷检查了。<br>Bad: 那个眼睛被玻璃珠检查了。 |
| 起飞 → 批评 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太起飞了。<br>Bad: 李太太批评了。 |
| 起飞 → 抨击 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士起飞了。<br>Bad: 宋女士抨击了。 |
| 起飞 → 拍摄 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七个哥哥起飞了。<br>Bad: 这七个哥哥拍摄了。 |
| 起飞 → 提醒 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四起飞了。<br>Bad: 李四提醒了。 |
| 起飞 → 支持 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们的兄弟起飞了。<br>Bad: 她们的兄弟支持了。 |
| 起飞 → 欺骗 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位舞者起飞了。<br>Bad: 那位舞者欺骗了。 |
| 起飞 → 清蒸 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她起飞了。<br>Bad: 她清蒸了。 |
| 起飞 → 爆炒 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位舞者起飞了。<br>Bad: 这位舞者爆炒了。 |
| 起飞 → 称赞 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的下属起飞了。<br>Bad: 他们的下属称赞了。 |
| 起飞 → 预习 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的上级起飞了。<br>Bad: 他们的上级预习了。 |
| 跑步 → 反感 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐跑步了。<br>Bad: 徐小姐反感了。 |
| 跑步 → 称赞 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你跑步了。<br>Bad: 你称赞了。 |
| 跑步 → 表扬 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那五位同事跑步了。<br>Bad: 那五位同事表扬了。 |
| 跨越 → 凝固 | agent_causative | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外三个老板跨越了海洋。<br>Bad: 另外三个老板凝固了海洋。 |
| 跨越 → 出现 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六位音乐家几乎跨越了海洋。<br>Bad: 这六位音乐家几乎出现了海洋。 |
| 跳舞 → 创作 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥跳舞了。<br>Bad: 杨大哥创作了。 |
| 跳舞 → 取代 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生的母亲跳舞了。<br>Bad: 李先生的母亲取代了。 |
| 跳舞 → 预习 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们跳舞了。<br>Bad: 你们预习了。 |
| 躺下 → 偷听 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三的同事躺下了。<br>Bad: 张三的同事偷听了。 |
| 躺下 → 前往 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥躺下了。<br>Bad: 冯大哥前往了。 |
| 躺下 → 厌恶 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们躺下了。<br>Bad: 你们厌恶了。 |
| 躺下 → 取缔 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个演奏员躺下了。<br>Bad: 这个演奏员取缔了。 |
| 躺下 → 爱戴 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶躺下了。<br>Bad: 张婶爱戴了。 |
| 躺下 → 爱护 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五位母亲躺下了。<br>Bad: 这五位母亲爱护了。 |
| 过去 → 反感 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶过去了。<br>Bad: 张婶反感了。 |
| 过去 → 反驳 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们过去了。<br>Bad: 他们反驳了。 |
| 过去 → 呵斥 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外八个奴隶过去了。<br>Bad: 另外八个奴隶呵斥了。 |
| 过去 → 喜欢 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位司机过去了。<br>Bad: 那位司机喜欢了。 |
| 过去 → 打断 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨过去了。<br>Bad: 王姨打断了。 |
| 过去 → 找到 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷过去了。<br>Bad: 胡大爷找到了。 |
| 过去 → 推崇 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那九个顾客过去了。<br>Bad: 那九个顾客推崇了。 |
| 过去 → 支持 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们过去了。<br>Bad: 她们支持了。 |
| 过去 → 演奏 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生过去了。<br>Bad: 刘先生演奏了。 |
| 过去 → 登上 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五个演奏员过去了。<br>Bad: 这五个演奏员登上了。 |
| 过去 → 观看 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她过去了。<br>Bad: 她观看了。 |
| 过去 → 领养 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五位记者过去了。<br>Bad: 这五位记者领养了。 |
| 过来 → 控制 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的妈妈过来了。<br>Bad: 我们的妈妈控制了。 |
| 过来 → 照顾 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位服务员过来了。<br>Bad: 那位服务员照顾了。 |
| 过来 → 表扬 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位演奏员过来了。<br>Bad: 这位演奏员表扬了。 |
| 运动 → 喜欢 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位领导运动了。<br>Bad: 这位领导喜欢了。 |
| 运动 → 登上 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们运动了。<br>Bad: 他们登上了。 |
| 运动 → 相信 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王的下属运动了。<br>Bad: 小王的下属相信了。 |
| 运动 → 知道 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位父亲运动了。<br>Bad: 这位父亲知道了。 |
| 运动 → 辩护 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那个消费者运动了。<br>Bad: 那个消费者辩护了。 |
| 运动 → 追捧 | intransitive_no_obj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太运动了。<br>Bad: 李太太追捧了。 |
| 郑大妈 → 收音机 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这八把椅子被郑大妈预习了。<br>Bad: 这八把椅子被收音机预习了。 |
| 钢琴家 → 收音机 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈的钢琴家清蒸了鸡。<br>Bad: 郑大妈的收音机清蒸了鸡。 |
| 钢琴家 → 电视剧 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 钢琴家麻醉过大象了。<br>Bad: 电视剧麻醉过大象了。 |
| 闲逛 → 憎恨 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他闲逛了。<br>Bad: 他憎恨了。 |
| 闲逛 → 爱护 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那五个吉他手闲逛了。<br>Bad: 那五个吉他手爱护了。 |
| 闲逛 → 登上 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外七位上级闲逛了。<br>Bad: 另外七位上级登上了。 |
| 闲逛 → 重建 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人闲逛了。<br>Bad: 张夫人重建了。 |
| 陈大姐 → 巧克力 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六只袜子被陈大姐盖了。<br>Bad: 这六只袜子被巧克力盖了。 |
| 陈大姐 → 开瓶器 | agent_animacy_passive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这五把椅子被陈大姐搬了。<br>Bad: 这五把椅子被开瓶器搬了。 |
| 陈大姐 → 方便面 | agent_animacy_passive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那一把椅子被陈大姐搬了。<br>Bad: 那一把椅子被方便面搬了。 |
| 音乐家 → 冰红茶 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三的音乐家清洗了杯子。<br>Bad: 张三的冰红茶清洗了杯子。 |
| 音乐家 → 录像带 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的音乐家创作了漫画。<br>Bad: 你们的录像带创作了漫画。 |
| 顾客 → 教材 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 顾客拉过小提琴了。<br>Bad: 教材拉过小提琴了。 |
| 顾客 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三的顾客屠宰过牛了。<br>Bad: 张三的杯子屠宰过牛了。 |
| 顾客 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 顾客吹过双簧了。<br>Bad: 袜子吹过双簧了。 |
| 顾客 → 裙子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐的顾客把鸭烧了。<br>Bad: 陈大姐的裙子把鸭烧了。 |
| 顾客 → 香蕉 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 顾客弹了玻璃珠。<br>Bad: 香蕉弹了玻璃珠。 |
| 领养 → 出现 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你差点儿领养了小猫。<br>Bad: 你差点儿出现了小猫。 |
| 领养 → 消失 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他领养了小猫。<br>Bad: 他消失了小猫。 |
| 领导 → 坚果 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们的领导盖了被子。<br>Bad: 你们的坚果盖了被子。 |
| 领导 → 杯子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 领导制作过电影了。<br>Bad: 杯子制作过电影了。 |
| 领导 → 红酒 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你的领导预习了教材。<br>Bad: 你的红酒预习了教材。 |
| 领导 → 衣服 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 领导喝了啤酒。<br>Bad: 衣服喝了啤酒。 |
| 领导 → 袜子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 领导跨越过沙漠了。<br>Bad: 袜子跨越过沙漠了。 |
| 领导 → 裤子 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她的领导喝了白酒。<br>Bad: 她的裤子喝了白酒。 |
| 领导 → 视频 | agent_animacy_subj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 领导领养过小狗了。<br>Bad: 视频领养过小狗了。 |
| 颤抖 → 制作 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那两个罪犯颤抖了。<br>Bad: 那两个罪犯制作了。 |
| 颤抖 → 前往 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位打工人颤抖了。<br>Bad: 这位打工人前往了。 |
| 颤抖 → 尊重 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这两个音乐家颤抖了。<br>Bad: 这两个音乐家尊重了。 |
| 颤抖 → 捕捉 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们颤抖了。<br>Bad: 你们捕捉了。 |
| 颤抖 → 相信 | intransitive_no_obj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位空姐颤抖了。<br>Bad: 这位空姐相信了。 |
| 飞行员 → 蛋炒饭 | agent_animacy_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我的飞行员吃过橘子了。<br>Bad: 我的蛋炒饭吃过橘子了。 |
| 驾驶 → 消失 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十个演奏员驾驶了飞机。<br>Bad: 另外十个演奏员消失了飞机。 |
| 麻醉 → 存在 | agent_causative | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她麻醉了老虎。<br>Bad: 她存在了老虎。 |

## classifier

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad inserts 个蛋糕是; 个蛋糕 -> 的 | classifier_noun_subj | 5 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们想拿个蛋糕。<br>Bad: 个蛋糕是他们想拿的。 |
| multiple edits: bad inserts 杯白酒是; 杯白酒 -> 的 | classifier_noun_subj | 5 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我想买杯白酒。<br>Bad: 杯白酒是我想买的。 |
| multiple edits: bad inserts 瓶红酒是; 瓶红酒 -> 的 | classifier_noun_subj | 4 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们想买瓶红酒。<br>Bad: 瓶红酒是我们想买的。 |
| multiple edits: bad deletes 你想买; bad inserts 是你想买的 | classifier_noun_subj | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你想买桶矿泉水。<br>Bad: 桶矿泉水是你想买的。 |
| multiple edits: bad inserts 杯红茶是; 杯红茶 -> 的 | classifier_noun_subj | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李四想买杯红茶。<br>Bad: 杯红茶是李四想买的。 |
| multiple edits: bad inserts 瓶矿泉水是; 瓶矿泉水 -> 的 | classifier_noun_subj | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们想要瓶矿泉水。<br>Bad: 瓶矿泉水是她们想要的。 |
| multiple edits: bad deletes 你想拿; bad inserts 是你想拿的 | classifier_noun_subj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你想拿个饮料瓶。<br>Bad: 个饮料瓶是你想拿的。 |
| multiple edits: bad inserts 张桌子是; 张桌子 -> 的 | classifier_noun_subj | 20 | 0.0000 | 0.9000 | -0.9000 | 0.0000 | Good: 王大娘想要张桌子。<br>Bad: 张桌子是王大娘想要的。 |
| multiple edits: bad inserts 杯红酒是; 杯红酒 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.8000 | -0.8000 | 0.0000 | Good: 他们想拿杯红酒。<br>Bad: 杯红酒是他们想拿的。 |
| multiple edits: bad inserts 把椅子是; 把椅子 -> 的 | classifier_noun_subj | 12 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 胡大爷想要把椅子。<br>Bad: 把椅子是胡大爷想要的。 |
| multiple edits: bad deletes 她想拿; bad inserts 是她想拿的 | classifier_noun_subj | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 她想拿桶矿泉水。<br>Bad: 桶矿泉水是她想拿的。 |
| multiple edits: bad inserts 瓶冰红茶是; 瓶冰红茶 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.6000 | -0.6000 | 0.0000 | Good: 徐小姐想拿瓶冰红茶。<br>Bad: 瓶冰红茶是徐小姐想拿的。 |
| multiple edits: bad inserts 瓶白酒是; 瓶白酒 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.6000 | -0.6000 | 0.0000 | Good: 我想买瓶白酒。<br>Bad: 瓶白酒是我想买的。 |
| multiple edits: bad inserts 条裤子是; 条裤子 -> 的 | classifier_noun_subj | 7 | 0.0000 | 0.5714 | -0.5714 | 0.0000 | Good: 冯大哥想拿条裤子。<br>Bad: 条裤子是冯大哥想拿的。 |
| 位 → 杯 | classifier_noun_agreement | 16 | 0.8125 | 0.3125 | +0.5000 | 0.0000 | Good: 这边坐着好几位演奏员。<br>Bad: 这边坐着好几杯演奏员。 |
| multiple edits: bad inserts 个充电器是; 个充电器 -> 的 | classifier_noun_subj | 6 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 王大娘想买个充电器。<br>Bad: 个充电器是王大娘想买的。 |
| multiple edits: bad deletes 我想拿; bad inserts 是我想拿的 | classifier_noun_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 我想拿个饮料瓶。<br>Bad: 个饮料瓶是我想拿的。 |
| multiple edits: bad inserts 个开瓶器是; 个开瓶器 -> 的 | classifier_noun_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 李太太想拿个开瓶器。<br>Bad: 个开瓶器是李太太想拿的。 |
| multiple edits: bad inserts 个橘子是; 个橘子 -> 的 | classifier_noun_subj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 李太太想买个橘子。<br>Bad: 个橘子是李太太想买的。 |
| multiple edits: bad inserts 条被子是; 条被子 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.4000 | -0.4000 | 0.0000 | Good: 陈大姐想买条被子。<br>Bad: 条被子是陈大姐想买的。 |
| multiple edits: bad inserts 桶方便面是; 桶方便面 -> 的 | classifier_noun_subj | 11 | 0.0000 | 0.3636 | -0.3636 | 0.0000 | Good: 你们想买桶方便面。<br>Bad: 桶方便面是你们想买的。 |
| 个 → 桶 | classifier_noun_agreement | 12 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 那边坐着十几个消防员。<br>Bad: 那边坐着十几桶消防员。 |
| 位 → 桶 | classifier_noun_agreement | 12 | 0.8333 | 0.5000 | +0.3333 | 0.0000 | Good: 那边站着三位服务员。<br>Bad: 那边站着三桶服务员。 |
| multiple edits: bad inserts 个收音机是; 个收音机 -> 的 | classifier_noun_subj | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 我们想要个收音机。<br>Bad: 个收音机是我们想要的。 |
| multiple edits: bad inserts 个馒头是; 个馒头 -> 的 | classifier_noun_subj | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 何太太想买个馒头。<br>Bad: 个馒头是何太太想买的。 |
| multiple edits: bad inserts 块蛋糕是; 块蛋糕 -> 的 | classifier_noun_subj | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 你们想拿块蛋糕。<br>Bad: 块蛋糕是你们想拿的。 |
| 位 → 串 | classifier_noun_agreement | 28 | 0.6429 | 0.3214 | +0.3214 | 0.0000 | Good: 这边坐着四位打工人。<br>Bad: 这边坐着四串打工人。 |
| 个 → 杯 | classifier_noun_agreement | 22 | 0.9091 | 0.5909 | +0.3182 | 0.0000 | Good: 这边躺着五个消费者。<br>Bad: 这边躺着五杯消费者。 |
| multiple edits: bad inserts 条裙子是; 条裙子 -> 的 | classifier_noun_subj | 7 | 0.0000 | 0.2857 | -0.2857 | 0.0000 | Good: 郑大妈想拿条裙子。<br>Bad: 条裙子是郑大妈想拿的。 |
| 位 → 块 | classifier_noun_agreement | 28 | 0.6429 | 0.3929 | +0.2500 | 0.0000 | Good: 那边躺着一位消费者。<br>Bad: 那边躺着一块消费者。 |
| multiple edits: bad inserts 个杯子是; 个杯子 -> 的 | classifier_noun_subj | 8 | 0.2500 | 0.5000 | -0.2500 | 0.0000 | Good: 你们想要个杯子。<br>Bad: 个杯子是你们想要的。 |
| multiple edits: bad inserts 桶矿泉水是; 桶矿泉水 -> 的 | classifier_noun_subj | 8 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 何太太想拿桶矿泉水。<br>Bad: 桶矿泉水是何太太想拿的。 |
| multiple edits: bad inserts 杯葡萄汁是; 杯葡萄汁 -> 的 | classifier_noun_subj | 4 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 你们想拿杯葡萄汁。<br>Bad: 杯葡萄汁是你们想拿的。 |
| multiple edits: bad inserts 瓶橙汁是; 瓶橙汁 -> 的 | classifier_noun_subj | 4 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 你想拿瓶橙汁。<br>Bad: 瓶橙汁是你想拿的。 |
| 头 → 位 | classifier_noun_agreement_no_gap | 47 | 0.4043 | 0.5957 | -0.1915 | 0.0000 | Good: 吴太太卖了好几百头钢琴家喜欢吃的大象。<br>Bad: 吴太太卖了好几百位钢琴家喜欢吃的大象。 |
| 个 → 串 | classifier_noun_agreement | 28 | 0.6429 | 0.5000 | +0.1429 | 0.0000 | Good: 那边坐着四个司机。<br>Bad: 那边坐着四串司机。 |
| 位 → 瓶 | classifier_noun_agreement | 14 | 0.6429 | 0.5000 | +0.1429 | 0.0000 | Good: 这边坐着三位消防员。<br>Bad: 这边坐着三瓶消防员。 |
| 位 → 只 | classifier_noun_agreement | 22 | 0.5000 | 0.3636 | +0.1364 | 0.0000 | Good: 那边站着七位同事。<br>Bad: 那边站着七只同事。 |
| 只 → 个 | classifier_noun_agreement_no_gap | 42 | 0.7857 | 0.6667 | +0.1190 | 0.0000 | Good: 我们卖了好几只记者喜欢吃的鸭。<br>Bad: 我们卖了好几个记者喜欢吃的鸭。 |
| 个 → 条 | classifier_noun_agreement | 18 | 0.4444 | 0.3333 | +0.1111 | 0.0000 | Good: 那边站着非常多个父亲。<br>Bad: 那边站着非常多条父亲。 |
| multiple edits: bad inserts 桶啤酒是; 桶啤酒 -> 的 | classifier_noun_subj | 23 | 0.0000 | 0.0870 | -0.0870 | 0.0000 | Good: 王五想拿桶啤酒。<br>Bad: 桶啤酒是王五想拿的。 |
| 位 → 条 | classifier_noun_agreement | 12 | 0.7500 | 0.6667 | +0.0833 | 0.0000 | Good: 那边躺着好几百位学生。<br>Bad: 那边躺着好几百条学生。 |
| 条 → 个 | classifier_noun_agreement_no_gap | 49 | 0.4082 | 0.4898 | -0.0816 | 0.0000 | Good: 你们买了五条姐妹喜欢吃的鱼。<br>Bad: 你们买了五个姐妹喜欢吃的鱼。 |
| 个 → 瓶 | classifier_noun_agreement | 13 | 0.6923 | 0.6154 | +0.0769 | 0.0000 | Good: 那边坐着两个服务员。<br>Bad: 那边坐着两瓶服务员。 |
| 位 → 头 | classifier_noun_agreement | 13 | 0.5385 | 0.4615 | +0.0769 | 0.0000 | Good: 那边站着四位钢琴家。<br>Bad: 那边站着四头钢琴家。 |
| 个 → 只 | classifier_noun_agreement | 14 | 0.4286 | 0.5000 | -0.0714 | 0.0000 | Good: 这边坐着好几百个钢琴家。<br>Bad: 这边坐着好几百只钢琴家。 |
| 个 → 头 | classifier_noun_agreement | 20 | 0.4500 | 0.5000 | -0.0500 | 0.0000 | Good: 那边坐着好几个演员。<br>Bad: 那边坐着好几头演员。 |
| 个 → 块 | classifier_noun_agreement | 28 | 0.5714 | 0.5357 | +0.0357 | 0.0000 | Good: 那边躺着一个打工人。<br>Bad: 那边躺着一块打工人。 |
| 头 → 个 | classifier_noun_agreement_no_gap | 61 | 0.6557 | 0.6230 | +0.0328 | 0.0000 | Good: 冯大哥卖了三头工人喜欢吃的牛。<br>Bad: 冯大哥卖了三个工人喜欢吃的牛。 |
| 条 → 位 | classifier_noun_agreement_no_gap | 51 | 0.3725 | 0.3529 | +0.0196 | 0.0000 | Good: 他们卖了好几百条罪犯喜欢吃的蛇。<br>Bad: 他们卖了好几百位罪犯喜欢吃的蛇。 |
| 只 → 位 | classifier_noun_agreement_no_gap | 50 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 她们买了几只顾客喜欢吃的鸭。<br>Bad: 她们买了几位顾客喜欢吃的鸭。 |
| multiple edits: bad inserts 片面包是; 片面包 -> 的 | classifier_noun_subj | 19 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人想拿片面包。<br>Bad: 片面包是张夫人想拿的。 |
| multiple edits: bad inserts 串香蕉是; 串香蕉 -> 的 | classifier_noun_subj | 14 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们想拿串香蕉。<br>Bad: 串香蕉是他们想拿的。 |
| multiple edits: bad inserts 只袜子是; 只袜子 -> 的 | classifier_noun_subj | 12 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她想拿只袜子。<br>Bad: 只袜子是她想拿的。 |
| multiple edits: bad inserts 本教材是; 本教材 -> 的 | classifier_noun_subj | 12 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她想拿本教材。<br>Bad: 本教材是她想拿的。 |
| multiple edits: bad inserts 只手套是; 只手套 -> 的 | classifier_noun_subj | 11 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们想买只手套。<br>Bad: 只手套是你们想买的。 |
| multiple edits: bad inserts 瓶可乐是; 瓶可乐 -> 的 | classifier_noun_subj | 8 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈想要瓶可乐。<br>Bad: 瓶可乐是周大妈想要的。 |
| multiple edits: bad inserts 块糖是; 块糖 -> 的 | classifier_noun_subj | 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生想买块糖。<br>Bad: 块糖是王先生想买的。 |
| multiple edits: bad inserts 杯咖啡是; 杯咖啡 -> 的 | classifier_noun_subj | 6 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我想要杯咖啡。<br>Bad: 杯咖啡是我想要的。 |
| multiple edits: bad inserts 个花卷是; 个花卷 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们想拿个花卷。<br>Bad: 个花卷是他们想拿的。 |
| multiple edits: bad inserts 个香蕉是; 个香蕉 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她想要个香蕉。<br>Bad: 个香蕉是她想要的。 |
| multiple edits: bad inserts 瓶啤酒是; 瓶啤酒 -> 的 | classifier_noun_subj | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他想拿瓶啤酒。<br>Bad: 瓶啤酒是他想拿的。 |
| multiple edits: bad inserts 块糖果是; 块糖果 -> 的 | classifier_noun_subj | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈想买块糖果。<br>Bad: 块糖果是周大妈想买的。 |
| multiple edits: bad inserts 杯橙汁是; 杯橙汁 -> 的 | classifier_noun_subj | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们想买杯橙汁。<br>Bad: 杯橙汁是他们想买的。 |
| multiple edits: bad inserts 杯牛奶是; 杯牛奶 -> 的 | classifier_noun_subj | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五想要杯牛奶。<br>Bad: 杯牛奶是王五想要的。 |
| multiple edits: bad deletes 他想要; bad inserts 是他想要的 | classifier_noun_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他想要个收音机。<br>Bad: 个收音机是他想要的。 |
| multiple edits: bad inserts 个苹果是; 个苹果 -> 的 | classifier_noun_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生想要个苹果。<br>Bad: 个苹果是李先生想要的。 |
| multiple edits: bad inserts 个鱼丸是; 个鱼丸 -> 的 | classifier_noun_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她想要个鱼丸。<br>Bad: 个鱼丸是她想要的。 |
| multiple edits: bad inserts 块巧克力是; 块巧克力 -> 的 | classifier_noun_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们想拿块巧克力。<br>Bad: 块巧克力是他们想拿的。 |
| multiple edits: bad deletes 他想拿; bad inserts 是他想拿的 | classifier_noun_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他想拿桶方便面。<br>Bad: 桶方便面是他想拿的。 |
| multiple edits: bad deletes 她想买; bad inserts 是她想买的 | classifier_noun_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她想买瓶冰红茶。<br>Bad: 瓶冰红茶是她想买的。 |
| multiple edits: bad deletes 她想要; bad inserts 是她想要的 | classifier_noun_subj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她想要个饮料瓶。<br>Bad: 个饮料瓶是她想要的。 |

## control_raising

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad deletes 那桶方便面; bad inserts 那桶方便面 | modal_raising_topicalization | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那桶方便面陈大姐应该能吃。<br>Bad: 陈大姐应该能那桶方便面吃。 |
| multiple edits: bad deletes 那部小说; bad inserts 那部小说 | modal_raising_topicalization | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那部小说王姨应该能创作。<br>Bad: 王姨应该能那部小说创作。 |
| 三 → 这个 | existential_there_subject_raising | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 有三个钢琴家入睡了。<br>Bad: 有这个个钢琴家入睡了。 |
| multiple edits: bad deletes 这本漫画; bad inserts 这本漫画 | modal_raising_topicalization | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这本漫画刘先生应该可以创作。<br>Bad: 刘先生应该可以这本漫画创作。 |
| multiple edits: bad deletes 这杯牛奶; bad inserts 这杯牛奶 | modal_raising_topicalization | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这杯牛奶你们应该可以买。<br>Bad: 你们应该可以这杯牛奶买。 |
| multiple edits: bad deletes 这杯白酒; bad inserts 这杯白酒 | modal_raising_topicalization | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这杯白酒王大娘应该能喝。<br>Bad: 王大娘应该能这杯白酒喝。 |
| multiple edits: bad deletes 这部视频; bad inserts 这部视频 | modal_raising_topicalization | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这部视频张夫人应该能制作。<br>Bad: 张夫人应该能这部视频制作。 |
| multiple edits: bad deletes 那杯红茶; bad inserts 那杯红茶 | modal_raising_topicalization | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那杯红茶我们应该会买。<br>Bad: 我们应该会那杯红茶买。 |
| multiple edits: bad deletes 那桶矿泉水; bad inserts 那桶矿泉水 | modal_raising_topicalization | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那桶矿泉水他们应该可以买。<br>Bad: 他们应该可以那桶矿泉水买。 |
| multiple edits: bad deletes 那部视频; bad inserts 那部视频 | modal_raising_topicalization | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那部视频小王应该能观看。<br>Bad: 小王应该能那部视频观看。 |
| multiple edits: bad inserts 李四应该会; bad deletes 李四应该会 | modal_raising_topicalization | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 那桶方便面李四应该会吃。<br>Bad: 李四应该会那桶方便面吃。 |
| 一 → 那些 | existential_there_subject_raising | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 有一个领导微笑了。<br>Bad: 有那些个领导微笑了。 |
| 十 → 那个 | existential_there_subject_raising | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 有十个领导看戏了。<br>Bad: 有那个个领导看戏了。 |
| multiple edits: bad deletes 那本书; bad inserts 那本书 | modal_raising_topicalization | 6 | 1.0000 | 0.1667 | +0.8333 | 0.0000 | Good: 那本书你们应该能看。<br>Bad: 你们应该能那本书看。 |
| multiple edits: bad deletes 这把椅子; bad inserts 这把椅子 | modal_raising_topicalization | 15 | 0.8667 | 0.0667 | +0.8000 | 0.0000 | Good: 这把椅子张先生应该可以搬。<br>Bad: 张先生应该可以这把椅子搬。 |
| multiple edits: bad deletes 那本手账; bad inserts 那本手账 | modal_raising_topicalization | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 那本手账她们应该能制作。<br>Bad: 她们应该能那本手账制作。 |
| multiple edits: bad deletes 那本教材; bad inserts 那本教材 | modal_raising_topicalization | 4 | 0.7500 | 0.0000 | +0.7500 | 0.0000 | Good: 那本教材你们应该可以预习。<br>Bad: 你们应该可以那本教材预习。 |
| multiple edits: bad deletes 那瓶橙汁; bad inserts 那瓶橙汁 | modal_raising_topicalization | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 那瓶橙汁你们应该可以喝。<br>Bad: 你们应该可以那瓶橙汁喝。 |
| 七 → 另外 | existential_there_subject_raising | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 有七位演员躺下了。<br>Bad: 有另外位演员躺下了。 |
| 六 → 那个 | existential_there_subject_raising | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 有六位老板坐下了。<br>Bad: 有那个位老板坐下了。 |
| multiple edits: bad deletes 那块蛋糕; bad inserts 那块蛋糕 | modal_raising_topicalization | 3 | 0.6667 | 0.0000 | +0.6667 | 0.0000 | Good: 那块蛋糕你应该可以买。<br>Bad: 你应该可以那块蛋糕买。 |
| 九 → 这个 | existential_there_subject_raising | 5 | 0.4000 | 1.0000 | -0.6000 | 0.0000 | Good: 有九个领导过去了。<br>Bad: 有这个个领导过去了。 |
| multiple edits: bad deletes 那把椅子; bad inserts 那把椅子 | modal_raising_topicalization | 12 | 0.6667 | 0.0833 | +0.5833 | 0.0000 | Good: 那把椅子李先生应该会搬。<br>Bad: 李先生应该会那把椅子搬。 |
| multiple edits: bad deletes 那块糖; bad inserts 那块糖 | modal_raising_topicalization | 7 | 0.8571 | 0.2857 | +0.5714 | 0.0000 | Good: 那块糖徐小姐应该能买。<br>Bad: 徐小姐应该能那块糖买。 |
| multiple edits: bad inserts 马上会; bad deletes 马上会 | modal_raising_hui | 23 | 0.8696 | 0.3043 | +0.5652 | 0.0000 | Good: 这杯白酒马上会变质。<br>Bad: 马上会这杯白酒变质。 |
| 三 → 那些 | existential_there_subject_raising | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 有三位打工人呼吸了。<br>Bad: 有那些位打工人呼吸了。 |
| multiple edits: bad deletes 这个玻璃珠; bad inserts 这个玻璃珠 | modal_raising_topicalization | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 这个玻璃珠张先生应该能弹。<br>Bad: 张先生应该能这个玻璃珠弹。 |
| multiple edits: bad deletes 这杯咖啡; bad inserts 这杯咖啡 | modal_raising_topicalization | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这杯咖啡我们应该会喝。<br>Bad: 我们应该会这杯咖啡喝。 |
| multiple edits: bad deletes 这瓶橙汁; bad inserts 这瓶橙汁 | modal_raising_topicalization | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这瓶橙汁刘先生应该能买。<br>Bad: 刘先生应该能这瓶橙汁买。 |
| multiple edits: bad deletes 那杯咖啡; bad inserts 那杯咖啡 | modal_raising_topicalization | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那杯咖啡张先生应该会买。<br>Bad: 张先生应该会那杯咖啡买。 |
| multiple edits: bad deletes 那杯牛奶; bad inserts 那杯牛奶 | modal_raising_topicalization | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 那杯牛奶他应该可以喝。<br>Bad: 他应该可以那杯牛奶喝。 |
| multiple edits: bad deletes 那瓶红酒; bad inserts 那瓶红酒 | modal_raising_topicalization | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 那瓶红酒她们应该能喝。<br>Bad: 她们应该能那瓶红酒喝。 |
| multiple edits: bad inserts 你应该可以; bad deletes 你应该可以 | modal_raising_topicalization | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 这个饮料瓶你应该可以卖。<br>Bad: 你应该可以这个饮料瓶卖。 |
| multiple edits: bad inserts 我应该能; bad deletes 我应该能 | modal_raising_topicalization | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那把椅子我应该能搬。<br>Bad: 我应该能那把椅子搬。 |
| 七 → 那些 | existential_there_subject_raising | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 有七位打工人哭了。<br>Bad: 有那些位打工人哭了。 |
| 三 → 这些 | existential_there_subject_raising | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 有三位上级躺下了。<br>Bad: 有这些位上级躺下了。 |
| 两 → 另外 | existential_there_subject_raising | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 有两个小孩溜走了。<br>Bad: 有另外个小孩溜走了。 |
| 两 → 这些 | existential_there_subject_raising | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 有两位工人健身了。<br>Bad: 有这些位工人健身了。 |
| 九 → 另外 | existential_there_subject_raising | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 有九个弟弟坐下了。<br>Bad: 有另外个弟弟坐下了。 |
| 九 → 这些 | existential_there_subject_raising | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 有九个消费者去了。<br>Bad: 有这些个消费者去了。 |
| 九 → 那些 | existential_there_subject_raising | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 有九个演员运动了。<br>Bad: 有那些个演员运动了。 |
| multiple edits: bad inserts 就要; bad deletes 就要 | modal_raising_hui | 30 | 0.8333 | 0.4000 | +0.4333 | 0.0000 | Good: 那杯白酒就要过期。<br>Bad: 就要那杯白酒过期。 |
| multiple edits: bad inserts 你应该会; bad deletes 你应该会 | modal_raising_topicalization | 7 | 0.5714 | 0.1429 | +0.4286 | 0.0000 | Good: 这片面包你应该会吃。<br>Bad: 你应该会这片面包吃。 |
| multiple edits: bad inserts 你应该能; bad deletes 你应该能 | modal_raising_topicalization | 5 | 0.8000 | 0.4000 | +0.4000 | 0.0000 | Good: 那瓶可乐你应该能买。<br>Bad: 你应该能那瓶可乐买。 |
| 四 → 这 | existential_there_subject_raising | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 有四个上级出发了。<br>Bad: 有这个上级出发了。 |
| multiple edits: bad inserts 他应该会; bad deletes 他应该会 | modal_raising_topicalization | 8 | 1.0000 | 0.6250 | +0.3750 | 0.0000 | Good: 那瓶白酒他应该会喝。<br>Bad: 他应该会那瓶白酒喝。 |
| 六 → 这个 | existential_there_subject_raising | 8 | 0.7500 | 0.3750 | +0.3750 | 0.0000 | Good: 有六个演奏员站立了。<br>Bad: 有这个个演奏员站立了。 |
| multiple edits: bad inserts 马上要; bad deletes 马上要 | modal_raising_hui | 28 | 0.6429 | 0.2857 | +0.3571 | 0.0000 | Good: 那块糖马上要过期。<br>Bad: 马上要那块糖过期。 |
| multiple edits: bad deletes 这条被子; bad inserts 这条被子 | modal_raising_topicalization | 18 | 0.3889 | 0.0556 | +0.3333 | 0.0000 | Good: 这条被子张夫人应该会盖。<br>Bad: 张夫人应该会这条被子盖。 |
| multiple edits: bad deletes 这片面包; bad inserts 这片面包 | modal_raising_topicalization | 9 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这片面包我们应该会吃。<br>Bad: 我们应该会这片面包吃。 |
| 六 → 这些 | existential_there_subject_raising | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 有六位顾客唱歌了。<br>Bad: 有这些位顾客唱歌了。 |
| multiple edits: bad deletes 这本手账; bad inserts 这本手账 | modal_raising_topicalization | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这本手账我们应该会制作。<br>Bad: 我们应该会这本手账制作。 |
| multiple edits: bad inserts 她们应该能; bad deletes 她们应该能 | modal_raising_topicalization | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 那部电视剧她们应该能制作。<br>Bad: 她们应该能那部电视剧制作。 |
| 一 → 这个 | existential_there_subject_raising | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 有一个记者微笑了。<br>Bad: 有这个个记者微笑了。 |
| 两 → 这个 | existential_there_subject_raising | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 有两位打工人爬行了。<br>Bad: 有这个位打工人爬行了。 |
| 四 → 那些 | existential_there_subject_raising | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 有四位顾客启程了。<br>Bad: 有那些位顾客启程了。 |
| 九 → 那个 | existential_there_subject_raising | 6 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 有九个小孩偷听了。<br>Bad: 有那个个小孩偷听了。 |
| multiple edits: bad deletes 那块糖果; bad inserts 那块糖果 | modal_raising_topicalization | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 那块糖果我应该可以买。<br>Bad: 我应该可以那块糖果买。 |
| 四 → 另外 | existential_there_subject_raising | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 有四个罪犯入睡了。<br>Bad: 有另外个罪犯入睡了。 |
| multiple edits: bad deletes 这本书; bad inserts 这本书 | modal_raising_topicalization | 7 | 1.0000 | 0.7143 | +0.2857 | 0.0000 | Good: 这本书刘先生应该会看。<br>Bad: 刘先生应该会这本书看。 |
| 八 → 那个 | existential_there_subject_raising | 7 | 0.8571 | 0.5714 | +0.2857 | 0.0000 | Good: 有八个音乐家叹息了。<br>Bad: 有那个个音乐家叹息了。 |
| multiple edits: bad inserts 要; bad deletes 要 | modal_raising_hui | 61 | 1.0000 | 0.7213 | +0.2787 | 0.0000 | Good: 那本小说要受潮。<br>Bad: 要那本小说受潮。 |
| multiple edits: bad deletes 那片面包; bad inserts 那片面包 | modal_raising_topicalization | 11 | 0.8182 | 0.5455 | +0.2727 | 0.0000 | Good: 那片面包你们应该能买。<br>Bad: 你们应该能那片面包买。 |
| multiple edits: bad deletes 那张桌子; bad inserts 那张桌子 | modal_raising_topicalization | 11 | 0.2727 | 0.0000 | +0.2727 | 0.0000 | Good: 那张桌子王大娘应该可以搬。<br>Bad: 王大娘应该可以那张桌子搬。 |
| multiple edits: bad deletes 那串香蕉; bad inserts 那串香蕉 | modal_raising_topicalization | 19 | 1.0000 | 0.7368 | +0.2632 | 0.0000 | Good: 那串香蕉张先生应该会吃。<br>Bad: 张先生应该会那串香蕉吃。 |
| multiple edits: bad inserts 他应该能; bad deletes 他应该能 | modal_raising_topicalization | 8 | 0.8750 | 0.6250 | +0.2500 | 0.0000 | Good: 那桶矿泉水他应该能喝。<br>Bad: 他应该能那桶矿泉水喝。 |
| 两 → 这 | existential_there_subject_raising | 8 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 有两个记者走了。<br>Bad: 有这个记者走了。 |
| multiple edits: bad deletes 那杯白酒; bad inserts 那杯白酒 | modal_raising_topicalization | 4 | 0.5000 | 0.7500 | -0.2500 | 0.0000 | Good: 那杯白酒杨大哥应该会喝。<br>Bad: 杨大哥应该会那杯白酒喝。 |
| 一 → 另外 | existential_there_subject_raising | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 有一个演员笑了。<br>Bad: 有另外个演员笑了。 |
| 三 → 这 | existential_there_subject_raising | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 有三位工人站立了。<br>Bad: 有这位工人站立了。 |
| 两 → 那些 | existential_there_subject_raising | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 有两位音乐家入睡了。<br>Bad: 有那些位音乐家入睡了。 |
| 八 → 这个 | existential_there_subject_raising | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 有八个领导躺下了。<br>Bad: 有这个个领导躺下了。 |
| 六 → 另外 | existential_there_subject_raising | 4 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 有六个罪犯呼吸了。<br>Bad: 有另外个罪犯呼吸了。 |
| 六 → 那些 | existential_there_subject_raising | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 有六个演奏员闲逛了。<br>Bad: 有那些个演奏员闲逛了。 |
| 十 → 另外 | existential_there_subject_raising | 4 | 0.5000 | 0.7500 | -0.2500 | 0.0000 | Good: 有十个顾客过来了。<br>Bad: 有另外个顾客过来了。 |
| 十 → 这个 | existential_there_subject_raising | 4 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 有十位服务员来了。<br>Bad: 有这个位服务员来了。 |
| 四 → 那 | existential_there_subject_raising | 4 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 有四位空姐品茶了。<br>Bad: 有那位空姐品茶了。 |
| multiple edits: bad inserts 就会; bad deletes 就会 | modal_raising_hui | 28 | 1.0000 | 0.7857 | +0.2143 | 0.0000 | Good: 那块糖就会熔化。<br>Bad: 就会那块糖熔化。 |
| 七 → 那个 | existential_there_subject_raising | 5 | 0.8000 | 0.6000 | +0.2000 | 0.0000 | Good: 有七个演员笑了。<br>Bad: 有那个个演员笑了。 |
| 三 → 那 | existential_there_subject_raising | 5 | 0.6000 | 0.4000 | +0.2000 | 0.0000 | Good: 有三位工人听课了。<br>Bad: 有那位工人听课了。 |
| 九 → 那 | existential_there_subject_raising | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 有九个舞者偷听了。<br>Bad: 有那个舞者偷听了。 |
| 两 → 那 | existential_there_subject_raising | 6 | 0.6667 | 0.8333 | -0.1667 | 0.0000 | Good: 有两个音乐家打架了。<br>Bad: 有那个音乐家打架了。 |
| 四 → 那个 | existential_there_subject_raising | 6 | 0.8333 | 0.6667 | +0.1667 | 0.0000 | Good: 有四位工人入睡了。<br>Bad: 有那个位工人入睡了。 |
| multiple edits: bad inserts 她应该会; bad deletes 她应该会 | modal_raising_topicalization | 6 | 0.5000 | 0.6667 | -0.1667 | 0.0000 | Good: 那桶啤酒她应该会喝。<br>Bad: 她应该会那桶啤酒喝。 |
| 七 → 这些 | existential_there_subject_raising | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 有七个罪犯运动了。<br>Bad: 有这些个罪犯运动了。 |
| 三 → 另外 | existential_there_subject_raising | 6 | 0.5000 | 0.6667 | -0.1667 | 0.0000 | Good: 有三个下属入睡了。<br>Bad: 有另外个下属入睡了。 |
| multiple edits: bad inserts 可能会; bad deletes 可能会 | modal_raising_hui | 32 | 0.9688 | 0.8125 | +0.1562 | 0.0000 | Good: 那块糖可能会熔化。<br>Bad: 可能会那块糖熔化。 |
| multiple edits: bad deletes 这块糖; bad inserts 这块糖 | modal_raising_topicalization | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 这块糖张三应该会吃。<br>Bad: 张三应该会这块糖吃。 |
| multiple edits: bad deletes 这部电影; bad inserts 这部电影 | modal_raising_topicalization | 7 | 0.8571 | 1.0000 | -0.1429 | 0.0000 | Good: 这部电影吴太太应该能拍摄。<br>Bad: 吴太太应该能这部电影拍摄。 |
| 九 → 这 | existential_there_subject_raising | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 有九个服务员打架了。<br>Bad: 有这个服务员打架了。 |
| 十 → 这 | existential_there_subject_raising | 7 | 0.8571 | 0.7143 | +0.1429 | 0.0000 | Good: 有十位母亲运动了。<br>Bad: 有这位母亲运动了。 |
| multiple edits: bad inserts 可能要; bad deletes 可能要 | modal_raising_hui | 30 | 1.0000 | 0.8667 | +0.1333 | 0.0000 | Good: 那片面包可能要变质。<br>Bad: 可能要那片面包变质。 |
| 六 → 这 | existential_there_subject_raising | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 有六位母亲睡觉了。<br>Bad: 有这位母亲睡觉了。 |
| 可以 → 希望 | control_modal_vs_raising_modal | 34 | 0.9706 | 0.8529 | +0.1176 | 0.0000 | Good: 另外六块糖可以热一些。<br>Bad: 另外六块糖希望热一些。 |
| 七 → 这 | existential_there_subject_raising | 9 | 1.0000 | 0.8889 | +0.1111 | 0.0000 | Good: 有七位演员出发了。<br>Bad: 有这位演员出发了。 |
| multiple edits: bad deletes 这张桌子; bad inserts 这张桌子 | modal_raising_topicalization | 9 | 0.3333 | 0.4444 | -0.1111 | 0.0000 | Good: 这张桌子我应该可以搬。<br>Bad: 我应该可以这张桌子搬。 |
| 应该 → 希望 | control_modal_vs_raising_modal | 32 | 0.9688 | 0.8750 | +0.0938 | 0.0000 | Good: 他们的另外八只手套应该昂贵一些。<br>Bad: 他们的另外八只手套希望昂贵一些。 |
| 应该 → 愿意 | control_modal_vs_raising_modal | 33 | 1.0000 | 0.9697 | +0.0303 | 0.0000 | Good: 她们的这六把椅子应该昂贵一些。<br>Bad: 她们的这六把椅子愿意昂贵一些。 |
| multiple edits: bad inserts 会; bad deletes 会 | modal_raising_hui | 68 | 1.0000 | 0.9706 | +0.0294 | 0.0000 | Good: 那块糖果会熔化。<br>Bad: 会那块糖果熔化。 |
| 应该 → 想要 | control_modal_vs_raising_modal | 45 | 0.9556 | 0.9778 | -0.0222 | 0.0000 | Good: 另外五本教材应该更便宜一点。<br>Bad: 另外五本教材想要更便宜一点。 |
| 可以 → 想要 | control_modal_vs_raising_modal | 51 | 0.9804 | 1.0000 | -0.0196 | 0.0000 | Good: 那九张桌子可以便宜一点。<br>Bad: 那九张桌子想要便宜一点。 |
| 可以 → 期待 | control_modal_vs_raising_modal | 39 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的另外六块巧克力可以鲜嫩一点。<br>Bad: 我们的另外六块巧克力期待鲜嫩一点。 |
| 可以 → 愿意 | control_modal_vs_raising_modal | 35 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这杯红茶可以更苦一点。<br>Bad: 这杯红茶愿意更苦一点。 |
| 应该 → 期待 | control_modal_vs_raising_modal | 31 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十张桌子应该更便宜一些。<br>Bad: 另外十张桌子期待更便宜一些。 |
| multiple edits: bad deletes 这串香蕉; bad inserts 这串香蕉 | modal_raising_topicalization | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这串香蕉她们应该会买。<br>Bad: 她们应该会这串香蕉买。 |
| 七 → 那 | existential_there_subject_raising | 8 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 有七位服务员坐下了。<br>Bad: 有那位服务员坐下了。 |
| 一 → 这 | existential_there_subject_raising | 7 | 0.8571 | 0.8571 | +0.0000 | 0.0000 | Good: 有一个钢琴家偷听了。<br>Bad: 有这个钢琴家偷听了。 |
| 一 → 那 | existential_there_subject_raising | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有一位演奏员站立了。<br>Bad: 有那位演奏员站立了。 |
| 五 → 这 | existential_there_subject_raising | 7 | 0.8571 | 0.8571 | +0.0000 | 0.0000 | Good: 有五个顾客来了。<br>Bad: 有这个顾客来了。 |
| multiple edits: bad deletes 那条被子; bad inserts 那条被子 | modal_raising_topicalization | 6 | 0.1667 | 0.1667 | +0.0000 | 0.0000 | Good: 那条被子他们应该可以盖。<br>Bad: 他们应该可以那条被子盖。 |
| 八 → 那 | existential_there_subject_raising | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有八位母亲运动了。<br>Bad: 有那位母亲运动了。 |
| multiple edits: bad deletes 那个杯子; bad inserts 那个杯子 | modal_raising_topicalization | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个杯子王姨应该能打碎。<br>Bad: 王姨应该能那个杯子打碎。 |
| 一 → 那个 | existential_there_subject_raising | 5 | 0.8000 | 0.8000 | +0.0000 | 0.0000 | Good: 有一位音乐家走了。<br>Bad: 有那个位音乐家走了。 |
| 两 → 那个 | existential_there_subject_raising | 5 | 0.8000 | 0.8000 | +0.0000 | 0.0000 | Good: 有两位领导微笑了。<br>Bad: 有那个位领导微笑了。 |
| 五 → 这些 | existential_there_subject_raising | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有五位演奏员微笑了。<br>Bad: 有这些位演奏员微笑了。 |
| 五 → 那 | existential_there_subject_raising | 5 | 0.8000 | 0.8000 | +0.0000 | 0.0000 | Good: 有五位上级颤抖了。<br>Bad: 有那位上级颤抖了。 |
| 十 → 那 | existential_there_subject_raising | 5 | 0.8000 | 0.8000 | +0.0000 | 0.0000 | Good: 有十位老师启程了。<br>Bad: 有那位老师启程了。 |
| multiple edits: bad deletes 这个杯子; bad inserts 这个杯子 | modal_raising_topicalization | 4 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 这个杯子张先生应该可以清洗。<br>Bad: 张先生应该可以这个杯子清洗。 |
| multiple edits: bad inserts 她应该能; bad deletes 她应该能 | modal_raising_topicalization | 4 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 那瓶可乐她应该能喝。<br>Bad: 她应该能那瓶可乐喝。 |
| multiple edits: bad inserts 我应该会; bad deletes 我应该会 | modal_raising_topicalization | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这本手账我应该会制作。<br>Bad: 我应该会这本手账制作。 |
| 五 → 另外 | existential_there_subject_raising | 4 | 0.2500 | 0.2500 | +0.0000 | 0.0000 | Good: 有五个哥哥呼吸了。<br>Bad: 有另外个哥哥呼吸了。 |
| 五 → 那个 | existential_there_subject_raising | 4 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 有五个演员闲逛了。<br>Bad: 有那个个演员闲逛了。 |
| 八 → 这 | existential_there_subject_raising | 4 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 有八个朋友看戏了。<br>Bad: 有这个朋友看戏了。 |
| 六 → 那 | existential_there_subject_raising | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有六位消费者呼吸了。<br>Bad: 有那位消费者呼吸了。 |
| 十 → 这些 | existential_there_subject_raising | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有十个顾客玩耍了。<br>Bad: 有这些个顾客玩耍了。 |
| 一 → 这些 | existential_there_subject_raising | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有一位员工出发了。<br>Bad: 有这些位员工出发了。 |
| 三 → 那个 | existential_there_subject_raising | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有三个顾客躺下了。<br>Bad: 有那个个顾客躺下了。 |
| 五 → 这个 | existential_there_subject_raising | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有五个奴隶哭了。<br>Bad: 有这个个奴隶哭了。 |
| 五 → 那些 | existential_there_subject_raising | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 有五位吉他手颤抖了。<br>Bad: 有那些位吉他手颤抖了。 |
| 八 → 另外 | existential_there_subject_raising | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有八位工人停下了。<br>Bad: 有另外位工人停下了。 |
| 八 → 那些 | existential_there_subject_raising | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有八个妹妹站立了。<br>Bad: 有那些个妹妹站立了。 |
| 四 → 这些 | existential_there_subject_raising | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有四位司机入睡了。<br>Bad: 有这些位司机入睡了。 |
| multiple edits: bad deletes 这个饮料瓶; bad inserts 这个饮料瓶 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个饮料瓶李四应该可以卖。<br>Bad: 李四应该可以这个饮料瓶卖。 |
| multiple edits: bad deletes 这块蛋糕; bad inserts 这块蛋糕 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这块蛋糕她们应该可以吃。<br>Bad: 她们应该可以这块蛋糕吃。 |
| multiple edits: bad deletes 这本小说; bad inserts 这本小说 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这本小说何太太应该可以创作。<br>Bad: 何太太应该可以这本小说创作。 |
| multiple edits: bad deletes 这瓶白酒; bad inserts 这瓶白酒 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这瓶白酒王小姐应该可以买。<br>Bad: 王小姐应该可以这瓶白酒买。 |
| multiple edits: bad deletes 那桶啤酒; bad inserts 那桶啤酒 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那桶啤酒王小姐应该能喝。<br>Bad: 王小姐应该能那桶啤酒喝。 |
| multiple edits: bad deletes 那瓶冰红茶; bad inserts 那瓶冰红茶 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那瓶冰红茶李太太应该能喝。<br>Bad: 李太太应该能那瓶冰红茶喝。 |
| multiple edits: bad deletes 那瓶可乐; bad inserts 那瓶可乐 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那瓶可乐王先生应该能喝。<br>Bad: 王先生应该能那瓶可乐喝。 |
| multiple edits: bad deletes 那部手账; bad inserts 那部手账 | modal_raising_topicalization | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那部手账王小姐应该能看。<br>Bad: 王小姐应该能那部手账看。 |
| multiple edits: bad deletes 那部漫画; bad inserts 那部漫画 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那部漫画她们应该能创作。<br>Bad: 她们应该能那部漫画创作。 |
| multiple edits: bad inserts 你们应该能; bad deletes 你们应该能 | modal_raising_topicalization | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 那个玻璃珠你们应该能弹。<br>Bad: 你们应该能那个玻璃珠弹。 |
| multiple edits: bad inserts 我应该可以; bad deletes 我应该可以 | modal_raising_topicalization | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这桶方便面我应该可以买。<br>Bad: 我应该可以这桶方便面买。 |
| 七 → 这个 | existential_there_subject_raising | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有七个打工人看戏了。<br>Bad: 有这个个打工人看戏了。 |
| 八 → 这些 | existential_there_subject_raising | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有八位舞者走路了。<br>Bad: 有这些位舞者走路了。 |
| 十 → 那些 | existential_there_subject_raising | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有十位员工睡觉了。<br>Bad: 有那些位员工睡觉了。 |
| 四 → 这个 | existential_there_subject_raising | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 有四个顾客站立了。<br>Bad: 有这个个顾客站立了。 |
| multiple edits: bad deletes 这个开瓶器; bad inserts 这个开瓶器 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个开瓶器冯大哥应该会买。<br>Bad: 冯大哥应该会这个开瓶器买。 |
| multiple edits: bad deletes 这块糖果; bad inserts 这块糖果 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这块糖果她们应该会买。<br>Bad: 她们应该会这块糖果买。 |
| multiple edits: bad deletes 这本教材; bad inserts 这本教材 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这本教材她们应该会预习。<br>Bad: 她们应该会这本教材预习。 |
| multiple edits: bad deletes 这杯红茶; bad inserts 这杯红茶 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这杯红茶他们应该能喝。<br>Bad: 他们应该能这杯红茶喝。 |
| multiple edits: bad deletes 这桶啤酒; bad inserts 这桶啤酒 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这桶啤酒他们应该会买。<br>Bad: 他们应该会这桶啤酒买。 |
| multiple edits: bad deletes 这桶方便面; bad inserts 这桶方便面 | modal_raising_topicalization | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这桶方便面何太太应该可以买。<br>Bad: 何太太应该可以这桶方便面买。 |
| multiple edits: bad deletes 这瓶啤酒; bad inserts 这瓶啤酒 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这瓶啤酒吴太太应该能喝。<br>Bad: 吴太太应该能这瓶啤酒喝。 |
| multiple edits: bad deletes 这部手账; bad inserts 这部手账 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这部手账我们应该能制作。<br>Bad: 我们应该能这部手账制作。 |
| multiple edits: bad deletes 这部漫画; bad inserts 这部漫画 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这部漫画你应该可以创作。<br>Bad: 你应该可以这部漫画创作。 |
| multiple edits: bad deletes 那个充电器; bad inserts 那个充电器 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个充电器张先生应该可以买。<br>Bad: 张先生应该可以那个充电器买。 |
| multiple edits: bad deletes 那个开瓶器; bad inserts 那个开瓶器 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个开瓶器张夫人应该会卖。<br>Bad: 张夫人应该会那个开瓶器卖。 |
| multiple edits: bad deletes 那个馒头; bad inserts 那个馒头 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个馒头李先生应该能吃。<br>Bad: 李先生应该能那个馒头吃。 |
| multiple edits: bad deletes 那块巧克力; bad inserts 那块巧克力 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那块巧克力杨大哥应该能买。<br>Bad: 杨大哥应该能那块巧克力买。 |
| multiple edits: bad deletes 那瓶啤酒; bad inserts 那瓶啤酒 | modal_raising_topicalization | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那瓶啤酒你应该可以喝。<br>Bad: 你应该可以那瓶啤酒喝。 |
| multiple edits: bad deletes 那瓶白酒; bad inserts 那瓶白酒 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那瓶白酒陈大姐应该能买。<br>Bad: 陈大姐应该能那瓶白酒买。 |
| multiple edits: bad deletes 那部电影; bad inserts 那部电影 | modal_raising_topicalization | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那部电影张先生应该会拍摄。<br>Bad: 张先生应该会那部电影拍摄。 |
| multiple edits: bad inserts 她们应该会; bad deletes 她们应该会 | modal_raising_topicalization | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个玻璃珠她们应该会弹。<br>Bad: 她们应该会这个玻璃珠弹。 |
| multiple edits: bad inserts 张婶应该能; bad deletes 张婶应该能 | modal_raising_topicalization | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那桶方便面张婶应该能吃。<br>Bad: 张婶应该能那桶方便面吃。 |

## ellipsis

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: 包扎 -> 入睡; 手 -> 一天; 包扎 -> 入睡 | ellipsis_adj | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他包扎了手，李先生也包扎了。<br>Bad: 他入睡了一天，李先生也入睡了。 |
| multiple edits: 烧 -> 唱歌; 鸭 -> 很久; 烧 -> 唱歌 | ellipsis_adj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们烧了鸭，你也烧了。<br>Bad: 他们唱歌了很久，你也唱歌了。 |
| multiple edits: 制作 -> 过来; 电视剧 -> 一小时; 制作 -> 过来 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张夫人制作了电视剧，杨大哥也制作了。<br>Bad: 张夫人过来了一小时，杨大哥也过来了。 |
| multiple edits: 包扎 -> 健身; 脚 -> 一分钟; 包扎 -> 健身 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈包扎了脚，吴太太也包扎了。<br>Bad: 周大妈健身了一分钟，吴太太也健身了。 |
| multiple edits: 包扎 -> 哭; 腿 -> 一天; 包扎 -> 哭 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你包扎了腿，你们也包扎了。<br>Bad: 你哭了一天，你们也哭了。 |
| multiple edits: 包扎 -> 微笑; 腿 -> 一会儿; 包扎 -> 微笑 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 郑大妈包扎了腿，我们也包扎了。<br>Bad: 郑大妈微笑了一会儿，我们也微笑了。 |
| multiple edits: 包扎 -> 玩耍; 腿 -> 一小时; 包扎 -> 玩耍 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们包扎了腿，他也包扎了。<br>Bad: 她们玩耍了一小时，他也玩耍了。 |
| multiple edits: 包扎 -> 睡觉; 鼻子 -> 一会儿; 包扎 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们包扎了鼻子，小王也包扎了。<br>Bad: 我们睡觉了一会儿，小王也睡觉了。 |
| multiple edits: 包扎 -> 颤抖; 手 -> 很久; 包扎 -> 颤抖 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他包扎了手，我们也包扎了。<br>Bad: 他颤抖了很久，我们也颤抖了。 |
| multiple edits: 吃 -> 听课; 蛋炒饭 -> 很久; 吃 -> 听课 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 宋女士吃了蛋炒饭，周大妈也吃了。<br>Bad: 宋女士听课了很久，周大妈也听课了。 |
| multiple edits: 吹 -> 健身; 双簧 -> 一分钟; 吹 -> 健身 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们吹了双簧，徐小姐也吹了。<br>Bad: 他们健身了一分钟，徐小姐也健身了。 |
| multiple edits: 吹 -> 睡觉; 双簧 -> 一分钟; 吹 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 宋女士吹了双簧，张夫人也吹了。<br>Bad: 宋女士睡觉了一分钟，张夫人也睡觉了。 |
| multiple edits: 吹 -> 站立; 笛子 -> 很久; 吹 -> 站立 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 陈大姐吹了笛子，王先生也吹了。<br>Bad: 陈大姐站立了很久，王先生也站立了。 |
| multiple edits: 唱 -> 偷听; bad inserts 一; 调 -> 时; 唱 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 郑大妈唱了小调，李先生也唱了。<br>Bad: 郑大妈偷听了一小时，李先生也偷听了。 |
| multiple edits: 唱 -> 看戏; 美声 -> 一天; 唱 -> 看戏 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李四唱了美声，她们也唱了。<br>Bad: 李四看戏了一天，她们也看戏了。 |
| multiple edits: 喝 -> 坐下; 牛奶 -> 一会儿; 喝 -> 坐下 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小明喝了牛奶，我也喝了。<br>Bad: 小明坐下了一会儿，我也坐下了。 |
| multiple edits: 喝 -> 笑; 可乐 -> 一分钟; 喝 -> 笑 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我喝了可乐，你也喝了。<br>Bad: 我笑了一分钟，你也笑了。 |
| multiple edits: 喝 -> 跑步; 白酒 -> 一天; 喝 -> 跑步 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 吴太太喝了白酒，王大娘也喝了。<br>Bad: 吴太太跑步了一天，王大娘也跑步了。 |
| multiple edits: 喝 -> 颤抖; 橙汁 -> 一小时; 喝 -> 颤抖 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们喝了橙汁，李四也喝了。<br>Bad: 我们颤抖了一小时，李四也颤抖了。 |
| multiple edits: 屠宰 -> 爬行; 牛 -> 一天; 屠宰 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐屠宰了牛，李先生也屠宰了。<br>Bad: 徐小姐爬行了一天，李先生也爬行了。 |
| multiple edits: 屠宰 -> 睡觉; 牛 -> 一会儿; 屠宰 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你屠宰了牛，王五也屠宰了。<br>Bad: 你睡觉了一会儿，王五也睡觉了。 |
| multiple edits: 弹 -> 坐下; 古筝 -> 一分钟; 弹 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她弹了古筝，周大妈也弹了。<br>Bad: 她坐下了一分钟，周大妈也坐下了。 |
| multiple edits: 弹 -> 起飞; 钢琴 -> 很久; 弹 -> 起飞 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小明弹了钢琴，你也弹了。<br>Bad: 小明起飞了很久，你也起飞了。 |
| multiple edits: 弹 -> 运动; 钢琴 -> 一分钟; 弹 -> 运动 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张婶弹了钢琴，他也弹了。<br>Bad: 张婶运动了一分钟，他也运动了。 |
| multiple edits: 打断 -> 健身; 脚 -> 一分钟; 打断 -> 健身 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们打断了脚，王五也打断了。<br>Bad: 她们健身了一分钟，王五也健身了。 |
| multiple edits: 拉 -> 出发; 大提琴 -> 很久; 拉 -> 出发 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你拉了大提琴，冯大哥也拉了。<br>Bad: 你出发了很久，冯大哥也出发了。 |
| multiple edits: 拉 -> 爬行; 小提琴 -> 一天; 拉 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 刘先生拉了小提琴，我也拉了。<br>Bad: 刘先生爬行了一天，我也爬行了。 |
| multiple edits: 拉 -> 站立; bad inserts 一; 提琴 -> 时; 拉 -> 站立 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太拉了小提琴，赵大爷也拉了。<br>Bad: 何太太站立了一小时，赵大爷也站立了。 |
| multiple edits: 拉 -> 笑; 大提琴 -> 一天; 拉 -> 笑 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生拉了大提琴，他们也拉了。<br>Bad: 刘先生笑了一天，他们也笑了。 |
| multiple edits: 拉 -> 颤抖; 大提琴 -> 一小时; 拉 -> 颤抖 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王小姐拉了大提琴，我们也拉了。<br>Bad: 王小姐颤抖了一小时，我们也颤抖了。 |
| multiple edits: 拍摄 -> 健身; 电影 -> 一天; 拍摄 -> 健身 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 宋女士拍摄了电影，陈大姐也拍摄了。<br>Bad: 宋女士健身了一天，陈大姐也健身了。 |
| multiple edits: 拍摄 -> 启程; 电影 -> 一会儿; 拍摄 -> 启程 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她拍摄了电影，王大娘也拍摄了。<br>Bad: 她启程了一会儿，王大娘也启程了。 |
| multiple edits: 检查 -> 健身; 心脏 -> 一小时; 检查 -> 健身 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张夫人检查了心脏，我们也检查了。<br>Bad: 张夫人健身了一小时，我们也健身了。 |
| multiple edits: 检查 -> 呼吸; 腿 -> 一分钟; 检查 -> 呼吸 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 吴太太检查了腿，他也检查了。<br>Bad: 吴太太呼吸了一分钟，他也呼吸了。 |
| multiple edits: 检查 -> 起飞; 肚子 -> 很久; 检查 -> 起飞 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王小姐检查了肚子，你也检查了。<br>Bad: 王小姐起飞了很久，你也起飞了。 |
| multiple edits: 检查 -> 起飞; 胃 -> 很久; 检查 -> 起飞 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生检查了胃，她们也检查了。<br>Bad: 李先生起飞了很久，她们也起飞了。 |
| multiple edits: 清洗 -> 呼吸; 杯子 -> 很久; 清洗 -> 呼吸 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐清洗了杯子，你也清洗了。<br>Bad: 徐小姐呼吸了很久，你也呼吸了。 |
| multiple edits: 清蒸 -> 游泳; 鸭 -> 很久; 清蒸 -> 游泳 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们清蒸了鸭，冯大哥也清蒸了。<br>Bad: 他们游泳了很久，冯大哥也游泳了。 |
| multiple edits: 清蒸 -> 爬行; 鸭 -> 很久; 清蒸 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她清蒸了鸭，我也清蒸了。<br>Bad: 她爬行了很久，我也爬行了。 |
| multiple edits: 演奏 -> 叹息; 狂想曲 -> 一分钟; 演奏 -> 叹息 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们演奏了狂想曲，小明也演奏了。<br>Bad: 你们叹息了一分钟，小明也叹息了。 |
| multiple edits: 演奏 -> 爬行; 奏鸣曲 -> 一会儿; 演奏 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他演奏了奏鸣曲，她也演奏了。<br>Bad: 他爬行了一会儿，她也爬行了。 |
| multiple edits: 演奏 -> 睡觉; 华尔兹 -> 一会儿; 演奏 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈演奏了华尔兹，小明也演奏了。<br>Bad: 周大妈睡觉了一会儿，小明也睡觉了。 |
| multiple edits: 演奏 -> 睡觉; 奏鸣曲 -> 一天; 演奏 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们演奏了奏鸣曲，你也演奏了。<br>Bad: 他们睡觉了一天，你也睡觉了。 |
| multiple edits: 演奏 -> 走路; 歌曲 -> 一小时; 演奏 -> 走路 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你演奏了歌曲，我们也演奏了。<br>Bad: 你走路了一小时，我们也走路了。 |
| multiple edits: 炖 -> 呼吸; 鱼 -> 一天; 炖 -> 呼吸 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们炖了鱼，周大妈也炖了。<br>Bad: 他们呼吸了一天，周大妈也呼吸了。 |
| multiple edits: 炖 -> 品茶; 鸭 -> 一天; 炖 -> 品茶 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她炖了鸭，他们也炖了。<br>Bad: 她品茶了一天，他们也品茶了。 |
| multiple edits: 炖 -> 坐下; 鸡 -> 一分钟; 炖 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你炖了鸡，张婶也炖了。<br>Bad: 你坐下了一分钟，张婶也坐下了。 |
| multiple edits: 炖 -> 颤抖; 鸭 -> 一小时; 炖 -> 颤抖 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们炖了鸭，小王也炖了。<br>Bad: 她们颤抖了一小时，小王也颤抖了。 |
| multiple edits: 烧 -> 唱歌; 鱼 -> 一天; 烧 -> 唱歌 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们烧了鱼，张夫人也烧了。<br>Bad: 我们唱歌了一天，张夫人也唱歌了。 |
| multiple edits: 烧 -> 唱歌; 鸡 -> 一分钟; 烧 -> 唱歌 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们烧了鸡，陈大姐也烧了。<br>Bad: 她们唱歌了一分钟，陈大姐也唱歌了。 |
| multiple edits: 煮 -> 呼吸; 鱼 -> 一天; 煮 -> 呼吸 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥煮了鱼，郑大妈也煮了。<br>Bad: 冯大哥呼吸了一天，郑大妈也呼吸了。 |
| multiple edits: 煮 -> 游泳; 鱼 -> 一天; 煮 -> 游泳 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五煮了鱼，冯大哥也煮了。<br>Bad: 王五游泳了一天，冯大哥也游泳了。 |
| multiple edits: 煮 -> 爬行; 鸡 -> 一天; 煮 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 杨大哥煮了鸡，何太太也煮了。<br>Bad: 杨大哥爬行了一天，何太太也爬行了。 |
| multiple edits: 爆炒 -> 微笑; 鸡 -> 一小时; 爆炒 -> 微笑 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生爆炒了鸡，她们也爆炒了。<br>Bad: 王先生微笑了一小时，她们也微笑了。 |
| multiple edits: 爆炒 -> 站立; 鸭 -> 一分钟; 爆炒 -> 站立 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李四爆炒了鸭，她也爆炒了。<br>Bad: 李四站立了一分钟，她也站立了。 |
| multiple edits: 爆炒 -> 笑; 鸡 -> 很久; 爆炒 -> 笑 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你爆炒了鸡，我们也爆炒了。<br>Bad: 你笑了很久，我们也笑了。 |
| multiple edits: 爆炒 -> 跑步; 鸭 -> 很久; 爆炒 -> 跑步 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王大娘爆炒了鸭，他也爆炒了。<br>Bad: 王大娘跑步了很久，他也跑步了。 |
| multiple edits: 盖 -> 游泳; 被子 -> 一会儿; 盖 -> 游泳 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们盖了被子，李先生也盖了。<br>Bad: 她们游泳了一会儿，李先生也游泳了。 |
| multiple edits: 盖 -> 跳舞; 被子 -> 一小时; 盖 -> 跳舞 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生盖了被子，你们也盖了。<br>Bad: 王先生跳舞了一小时，你们也跳舞了。 |
| multiple edits: 看 -> 跑步; 漫画 -> 一分钟; 看 -> 跑步 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们看了漫画，她也看了。<br>Bad: 他们跑步了一分钟，她也跑步了。 |
| multiple edits: 观看 -> 跳舞; 电影 -> 一会儿; 观看 -> 跳舞 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 郑大妈观看了电影，赵大爷也观看了。<br>Bad: 郑大妈跳舞了一会儿，赵大爷也跳舞了。 |
| multiple edits: 跨越 -> 健身; 沙漠 -> 一小时; 跨越 -> 健身 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我跨越了沙漠，她也跨越了。<br>Bad: 我健身了一小时，她也健身了。 |
| multiple edits: 跨越 -> 爬行; 海洋 -> 一小时; 跨越 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王先生跨越了海洋，他们也跨越了。<br>Bad: 王先生爬行了一小时，他们也爬行了。 |
| multiple edits: 预习 -> 哭; 教材 -> 一分钟; 预习 -> 哭 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他预习了教材，张婶也预习了。<br>Bad: 他哭了一分钟，张婶也哭了。 |
| multiple edits: 预习 -> 站立; 教材 -> 一分钟; 预习 -> 站立 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 周大妈预习了教材，冯大哥也预习了。<br>Bad: 周大妈站立了一分钟，冯大哥也站立了。 |
| multiple edits: 领养 -> 走路; 小猫 -> 一会儿; 领养 -> 走路 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们领养了小猫，张婶也领养了。<br>Bad: 你们走路了一会儿，张婶也走路了。 |
| multiple edits: 驾驶 -> 停下; 火车 -> 很久; 驾驶 -> 停下 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们驾驶了火车，李先生也驾驶了。<br>Bad: 他们停下了很久，李先生也停下了。 |
| multiple edits: 驾驶 -> 偷听; 货车 -> 一小时; 驾驶 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张三驾驶了货车，刘先生也驾驶了。<br>Bad: 张三偷听了一小时，刘先生也偷听了。 |
| multiple edits: 驾驶 -> 哭; 轮船 -> 一分钟; 驾驶 -> 哭 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李四驾驶了轮船，徐小姐也驾驶了。<br>Bad: 李四哭了一分钟，徐小姐也哭了。 |
| multiple edits: 驾驶 -> 站立; 轮船 -> 一会儿; 驾驶 -> 站立 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你驾驶了轮船，他也驾驶了。<br>Bad: 你站立了一会儿，他也站立了。 |
| multiple edits: 驾驶 -> 过来; 火车 -> 一小时; 驾驶 -> 过来 | ellipsis_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你驾驶了火车，王先生也驾驶了。<br>Bad: 你过来了一小时，王先生也过来了。 |
| multiple edits: 麻醉 -> 入睡; 老虎 -> 一会儿; 麻醉 -> 入睡 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太麻醉了老虎，王五也麻醉了。<br>Bad: 何太太入睡了一会儿，王五也入睡了。 |
| multiple edits: 麻醉 -> 出发; 大象 -> 一分钟; 麻醉 -> 出发 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王先生麻醉了大象，我也麻醉了。<br>Bad: 王先生出发了一分钟，我也出发了。 |
| multiple edits: 麻醉 -> 去; 老虎 -> 一天; 麻醉 -> 去 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 赵大爷麻醉了老虎，王小姐也麻醉了。<br>Bad: 赵大爷去了一天，王小姐也去了。 |
| multiple edits: 麻醉 -> 唱歌; 老虎 -> 很久; 麻醉 -> 唱歌 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 杨大哥麻醉了老虎，他也麻醉了。<br>Bad: 杨大哥唱歌了很久，他也唱歌了。 |
| multiple edits: 麻醉 -> 笑; 老虎 -> 一会儿; 麻醉 -> 笑 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们麻醉了老虎，刘先生也麻醉了。<br>Bad: 我们笑了一会儿，刘先生也笑了。 |
| multiple edits: 麻醉 -> 跳舞; 老虎 -> 一天; 麻醉 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们麻醉了老虎，张先生也麻醉了。<br>Bad: 她们跳舞了一天，张先生也跳舞了。 |
| multiple edits: 麻醉 -> 过去; 老虎 -> 一分钟; 麻醉 -> 过去 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小明麻醉了老虎，他也麻醉了。<br>Bad: 小明过去了一分钟，他也过去了。 |
| multiple edits: 麻醉 -> 过来; 大象 -> 一会儿; 麻醉 -> 过来 | ellipsis_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 刘先生麻醉了大象，你也麻醉了。<br>Bad: 刘先生过来了一会儿，你也过来了。 |
| 个 → 片 | ellipsis_n_bar_class | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太吃了七个花卷，周大妈十个。<br>Bad: 何太太吃了七个花卷，周大妈十片。 |
| 只 → 片 | ellipsis_n_bar_class | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 刘先生吃了三只鸡，郑大妈两只。<br>Bad: 刘先生吃了三只鸡，郑大妈两片。 |
| 头 → 条 | ellipsis_n_bar_class | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐捕捉了三头大象，张婶五头。<br>Bad: 陈大姐捕捉了三头大象，张婶五条。 |
| 桶 → 个 | ellipsis_n_bar_class | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李四吃了三桶方便面，张婶十桶。<br>Bad: 李四吃了三桶方便面，张婶十个。 |
| 桶 → 杯 | ellipsis_n_bar_class | 8 | 0.3750 | 0.8750 | -0.5000 | 0.0000 | Good: 王大娘喝了一桶矿泉水，张先生六桶。<br>Bad: 王大娘喝了一桶矿泉水，张先生六杯。 |
| 个 → 条 | ellipsis_n_bar_class | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 陈大姐检查了一个眼睛，王五四个。<br>Bad: 陈大姐检查了一个眼睛，王五四条。 |
| 条 → 头 | ellipsis_n_bar_class | 5 | 0.2000 | 0.6000 | -0.4000 | 0.0000 | Good: 李先生捕捉了四条蛇，赵大爷十条。<br>Bad: 李先生捕捉了四条蛇，赵大爷十头。 |
| 头 → 只 | ellipsis_n_bar_class | 16 | 0.3750 | 0.6250 | -0.2500 | 0.0000 | Good: 何太太麻醉了十头大象，陈大姐七头。<br>Bad: 何太太麻醉了十头大象，陈大姐七只。 |
| 瓶 → 桶 | ellipsis_n_bar_class | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 宋女士喝了两瓶橙汁，张三七瓶。<br>Bad: 宋女士喝了两瓶橙汁，张三七桶。 |
| 把 → 张 | ellipsis_n_bar_class | 13 | 0.3077 | 0.5385 | -0.2308 | 0.0000 | Good: 吴太太搬了五把椅子，王先生七把。<br>Bad: 吴太太搬了五把椅子，王先生七张。 |
| 瓶 → 杯 | ellipsis_n_bar_class | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 小明喝了十瓶红酒，王五六瓶。<br>Bad: 小明喝了十瓶红酒，王五六杯。 |
| 只 → 头 | ellipsis_n_bar_class | 17 | 0.7647 | 0.9412 | -0.1765 | 0.0000 | Good: 王先生麻醉了一只老虎，小王七只。<br>Bad: 王先生麻醉了一只老虎，小王七头。 |
| 只 → 条 | ellipsis_n_bar_class | 67 | 0.9254 | 0.7612 | +0.1642 | 0.0000 | Good: 胡大爷清蒸了四只鸭，杨大哥八只。<br>Bad: 胡大爷清蒸了四只鸭，杨大哥八条。 |
| 部 → 本 | ellipsis_n_bar_class | 25 | 0.6400 | 0.4800 | +0.1600 | 0.0000 | Good: 冯大哥看了七部日记，何太太四部。<br>Bad: 冯大哥看了七部日记，何太太四本。 |
| 条 → 只 | ellipsis_n_bar_class | 69 | 0.4928 | 0.5797 | -0.0870 | 0.0000 | Good: 王五领养了一条小狗，李四九条。<br>Bad: 王五领养了一条小狗，李四九只。 |
| 本 → 部 | ellipsis_n_bar_class | 25 | 0.8800 | 0.8400 | +0.0400 | 0.0000 | Good: 赵大爷制作了六本手账，杨大哥九本。<br>Bad: 赵大爷制作了六本手账，杨大哥九部。 |
| 张 → 把 | ellipsis_n_bar_class | 14 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈搬了四张桌子，张先生三张。<br>Bad: 郑大妈搬了四张桌子，张先生三把。 |
| 个 → 只 | ellipsis_n_bar_class | 7 | 0.1429 | 0.1429 | +0.0000 | 0.0000 | Good: 赵大爷检查了六个头，李四四个。<br>Bad: 赵大爷检查了六个头，李四四只。 |
| 是 → 卖给了王五 | ellipsis_double_object | 6 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们卖给了王五十几把椅子，那个上级也是。<br>Bad: 我们卖给了王五十几把椅子，那个上级也卖给了王五。 |
| 是 → 递给了李四 | ellipsis_double_object | 6 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你递给了李四许多个充电器，胡大爷的姐妹也是。<br>Bad: 你递给了李四许多个充电器，胡大爷的姐妹也递给了李四。 |
| 是 → 借给了张先生 | ellipsis_double_object | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们借给了张先生非常多张桌子，这十个姐姐也是。<br>Bad: 她们借给了张先生非常多张桌子，这十个姐姐也借给了张先生。 |
| 是 → 借给了徐小姐 | ellipsis_double_object | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐借给了徐小姐好几十张桌子，这四个上级也是。<br>Bad: 陈大姐借给了徐小姐好几十张桌子，这四个上级也借给了徐小姐。 |
| 是 → 卖给了冯大哥 | ellipsis_double_object | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你卖给了冯大哥一个收音机，王小姐的上级也是。<br>Bad: 你卖给了冯大哥一个收音机，王小姐的上级也卖给了冯大哥。 |
| 是 → 卖给了王姨 | ellipsis_double_object | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生卖给了王姨两个饮料瓶，另外九个服务员也是。<br>Bad: 张先生卖给了王姨两个饮料瓶，另外九个服务员也卖给了王姨。 |
| 是 → 寄给了刘先生 | ellipsis_double_object | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们寄给了刘先生三只手套，那两个弟弟也是。<br>Bad: 他们寄给了刘先生三只手套，那两个弟弟也寄给了刘先生。 |
| 是 → 送给了何太太 | ellipsis_double_object | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三送给了何太太好几十只手套，另外六位司机也是。<br>Bad: 张三送给了何太太好几十只手套，另外六位司机也送给了何太太。 |
| 条 → 个 | ellipsis_n_bar_class | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐包扎了五条腿，王小姐九条。<br>Bad: 陈大姐包扎了五条腿，王小姐九个。 |
| 是 → 买给了周大妈 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人买给了周大妈七条裤子，那个司机也是。<br>Bad: 张夫人买给了周大妈七条裤子，那个司机也买给了周大妈。 |
| 是 → 买给了张三 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘买给了张三两本教材，他的妈妈也是。<br>Bad: 王大娘买给了张三两本教材，他的妈妈也买给了张三。 |
| 是 → 借给了刘先生 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐借给了刘先生九本教材，王大娘的老板也是。<br>Bad: 徐小姐借给了刘先生九本教材，王大娘的老板也借给了刘先生。 |
| 是 → 借给了周大妈 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我借给了周大妈非常多个开瓶器，这个领导也是。<br>Bad: 我借给了周大妈非常多个开瓶器，这个领导也借给了周大妈。 |
| 是 → 借给了小王 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你借给了小王好几十把椅子，这位老板也是。<br>Bad: 你借给了小王好几十把椅子，这位老板也借给了小王。 |
| 是 → 借给了张三 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生借给了张三非常多条裤子，你的兄弟也是。<br>Bad: 张先生借给了张三非常多条裤子，你的兄弟也借给了张三。 |
| 是 → 借给了李太太 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太借给了李太太许多条裙子，这位钢琴家也是。<br>Bad: 吴太太借给了李太太许多条裙子，这位钢琴家也借给了李太太。 |
| 是 → 寄给了李先生 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈寄给了李先生九个开瓶器，这位母亲也是。<br>Bad: 周大妈寄给了李先生九个开瓶器，这位母亲也寄给了李先生。 |
| 是 → 寄给了李太太 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们寄给了李太太好几十个玻璃珠，你们的同事也是。<br>Bad: 她们寄给了李太太好几十个玻璃珠，你们的同事也寄给了李太太。 |
| 是 → 寄给了王先生 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐寄给了王先生五把椅子，另外两个下属也是。<br>Bad: 徐小姐寄给了王先生五把椅子，另外两个下属也寄给了王先生。 |
| 是 → 寄给了郑大妈 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们寄给了郑大妈三只手套，那个姐姐也是。<br>Bad: 你们寄给了郑大妈三只手套，那个姐姐也寄给了郑大妈。 |
| 是 → 送给了冯大哥 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我送给了冯大哥七把椅子，这个哥哥也是。<br>Bad: 我送给了冯大哥七把椅子，这个哥哥也送给了冯大哥。 |
| 是 → 送给了吴太太 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们送给了吴太太七条裙子，我的老板也是。<br>Bad: 我们送给了吴太太七条裙子，我的老板也送给了吴太太。 |
| 是 → 送给了小王 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他送给了小王几张桌子，那个儿子也是。<br>Bad: 他送给了小王几张桌子，那个儿子也送给了小王。 |
| 是 → 送给了李四 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他送给了李四三张桌子，那个打工人也是。<br>Bad: 他送给了李四三张桌子，那个打工人也送给了李四。 |
| 是 → 送给了王姨 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥送给了王姨七本教材，那四个儿子也是。<br>Bad: 冯大哥送给了王姨七本教材，那四个儿子也送给了王姨。 |
| 是 → 递给了周大妈 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她递给了周大妈非常多张桌子，你的上级也是。<br>Bad: 她递给了周大妈非常多张桌子，你的上级也递给了周大妈。 |
| 是 → 递给了王先生 | ellipsis_double_object | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥递给了王先生四张桌子，另外两位上级也是。<br>Bad: 冯大哥递给了王先生四张桌子，另外两位上级也递给了王先生。 |
| 只 → 个 | ellipsis_n_bar_class | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷包扎了四只手，刘先生九只。<br>Bad: 赵大爷包扎了四只手，刘先生九个。 |
| 是 → 买给了刘先生 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我买给了刘先生几条裤子，王先生的弟弟也是。<br>Bad: 我买给了刘先生几条裤子，王先生的弟弟也买给了刘先生。 |
| 是 → 借给了何太太 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她借给了何太太一个充电器，那位领导也是。<br>Bad: 她借给了何太太一个充电器，那位领导也借给了何太太。 |
| 是 → 借给了李四 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐借给了李四许多条裤子，徐小姐的父亲也是。<br>Bad: 王小姐借给了李四许多条裤子，徐小姐的父亲也借给了李四。 |
| 是 → 借给了王姨 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她借给了王姨几条裤子，她的母亲也是。<br>Bad: 她借给了王姨几条裤子，她的母亲也借给了王姨。 |
| 是 → 借给了赵大爷 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们借给了赵大爷两把椅子，这四个领导也是。<br>Bad: 我们借给了赵大爷两把椅子，这四个领导也借给了赵大爷。 |
| 是 → 借给了郑大妈 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐借给了郑大妈四个充电器，我的上级也是。<br>Bad: 徐小姐借给了郑大妈四个充电器，我的上级也借给了郑大妈。 |
| 是 → 借给了陈大姐 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他借给了陈大姐八把椅子，王五的老板也是。<br>Bad: 他借给了陈大姐八把椅子，王五的老板也借给了陈大姐。 |
| 是 → 卖给了刘先生 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我卖给了刘先生十条被子，另外八位服务员也是。<br>Bad: 我卖给了刘先生十条被子，另外八位服务员也卖给了刘先生。 |
| 是 → 卖给了吴太太 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈卖给了吴太太几个开瓶器，他的下属也是。<br>Bad: 郑大妈卖给了吴太太几个开瓶器，他的下属也卖给了吴太太。 |
| 是 → 卖给了周大妈 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明卖给了周大妈两本教材，你的女儿也是。<br>Bad: 小明卖给了周大妈两本教材，你的女儿也卖给了周大妈。 |
| 是 → 卖给了小王 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你卖给了小王十张桌子，那个女儿也是。<br>Bad: 你卖给了小王十张桌子，那个女儿也卖给了小王。 |
| 是 → 卖给了张夫人 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四卖给了张夫人许多个杯子，那个下属也是。<br>Bad: 李四卖给了张夫人许多个杯子，那个下属也卖给了张夫人。 |
| 是 → 卖给了张婶 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们卖给了张婶好几十条裤子，我的学生也是。<br>Bad: 你们卖给了张婶好几十条裤子，我的学生也卖给了张婶。 |
| 是 → 卖给了李太太 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她卖给了李太太五个收音机，那三位领导也是。<br>Bad: 她卖给了李太太五个收音机，那三位领导也卖给了李太太。 |
| 是 → 卖给了王小姐 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨卖给了王小姐六只手套，王姨的朋友也是。<br>Bad: 王姨卖给了王小姐六只手套，王姨的朋友也卖给了王小姐。 |
| 是 → 卖给了赵大爷 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐卖给了赵大爷七张桌子，他的下属也是。<br>Bad: 陈大姐卖给了赵大爷七张桌子，他的下属也卖给了赵大爷。 |
| 是 → 寄给了小明 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太寄给了小明九条被子，李四的学生也是。<br>Bad: 李太太寄给了小明九条被子，李四的学生也寄给了小明。 |
| 是 → 寄给了杨大哥 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们寄给了杨大哥七条裤子，我们的姐姐也是。<br>Bad: 我们寄给了杨大哥七条裤子，我们的姐姐也寄给了杨大哥。 |
| 是 → 寄给了王五 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太寄给了王五四张桌子，另外三位母亲也是。<br>Bad: 李太太寄给了王五四张桌子，另外三位母亲也寄给了王五。 |
| 是 → 送给了张先生 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我送给了张先生八张桌子，陈大姐的员工也是。<br>Bad: 我送给了张先生八张桌子，陈大姐的员工也送给了张先生。 |
| 是 → 送给了张婶 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生送给了张婶非常多本教材，我们的妈妈也是。<br>Bad: 张先生送给了张婶非常多本教材，我们的妈妈也送给了张婶。 |
| 是 → 送给了王五 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们送给了王五好几条裙子，另外九个顾客也是。<br>Bad: 她们送给了王五好几条裙子，另外九个顾客也送给了王五。 |
| 是 → 送给了郑大妈 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五送给了郑大妈五只手套，这七位吉他手也是。<br>Bad: 王五送给了郑大妈五只手套，这七位吉他手也送给了郑大妈。 |
| 是 → 递给了刘先生 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐递给了刘先生六只手套，这位记者也是。<br>Bad: 徐小姐递给了刘先生六只手套，这位记者也递给了刘先生。 |
| 是 → 递给了胡大爷 | ellipsis_double_object | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们递给了胡大爷好几百个玻璃珠，那五位下属也是。<br>Bad: 我们递给了胡大爷好几百个玻璃珠，那五位下属也递给了胡大爷。 |
| 杯 → 桶 | ellipsis_n_bar_class | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐喝了五杯橙汁，王小姐六杯。<br>Bad: 徐小姐喝了五杯橙汁，王小姐六桶。 |
| multiple edits: 观看 -> 闲逛; 电视剧 -> 一小时; 观看 -> 闲逛 | ellipsis_adj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我观看了电视剧，我们也观看了。<br>Bad: 我闲逛了一小时，我们也闲逛了。 |
| multiple edits: 预习 -> 走; 教材 -> 一小时; 预习 -> 走 | ellipsis_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五预习了教材，冯大哥也预习了。<br>Bad: 王五走了一小时，冯大哥也走了。 |
| 是 → 买给了何太太 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你买给了何太太七只手套，这三位领导也是。<br>Bad: 你买给了何太太七只手套，这三位领导也买给了何太太。 |
| 是 → 买给了小王 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你买给了小王几只手套，张三的老板也是。<br>Bad: 你买给了小王几只手套，张三的老板也买给了小王。 |
| 是 → 买给了张先生 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士买给了张先生好几本教材，那位记者也是。<br>Bad: 宋女士买给了张先生好几本教材，那位记者也买给了张先生。 |
| 是 → 买给了张夫人 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们买给了张夫人七把椅子，李太太的女儿也是。<br>Bad: 她们买给了张夫人七把椅子，李太太的女儿也买给了张夫人。 |
| 是 → 买给了王姨 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五买给了王姨三只袜子，这四位顾客也是。<br>Bad: 王五买给了王姨三只袜子，这四位顾客也买给了王姨。 |
| 是 → 买给了胡大爷 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太买给了胡大爷九张桌子，另外三个弟弟也是。<br>Bad: 吴太太买给了胡大爷九张桌子，另外三个弟弟也买给了胡大爷。 |
| 是 → 买给了赵大爷 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们买给了赵大爷八张桌子，这两位学生也是。<br>Bad: 我们买给了赵大爷八张桌子，这两位学生也买给了赵大爷。 |
| 是 → 借给了冯大哥 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们借给了冯大哥几条裙子，那位下属也是。<br>Bad: 我们借给了冯大哥几条裙子，那位下属也借给了冯大哥。 |
| 是 → 借给了吴太太 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我借给了吴太太五个充电器，张先生的妹妹也是。<br>Bad: 我借给了吴太太五个充电器，张先生的妹妹也借给了吴太太。 |
| 是 → 借给了宋女士 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们借给了宋女士许多张桌子，何太太的儿子也是。<br>Bad: 你们借给了宋女士许多张桌子，何太太的儿子也借给了宋女士。 |
| 是 → 借给了张夫人 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太借给了张夫人好几十张桌子，这个小孩也是。<br>Bad: 何太太借给了张夫人好几十张桌子，这个小孩也借给了张夫人。 |
| 是 → 借给了李先生 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她借给了李先生许多只手套，小明的朋友也是。<br>Bad: 她借给了李先生许多只手套，小明的朋友也借给了李先生。 |
| 是 → 卖给了李四 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨卖给了李四九条裙子，另外九个演员也是。<br>Bad: 王姨卖给了李四九条裙子，另外九个演员也卖给了李四。 |
| 是 → 卖给了郑大妈 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生卖给了郑大妈四张桌子，那位下属也是。<br>Bad: 王先生卖给了郑大妈四张桌子，那位下属也卖给了郑大妈。 |
| 是 → 寄给了冯大哥 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他寄给了冯大哥十个玻璃珠，另外八个罪犯也是。<br>Bad: 他寄给了冯大哥十个玻璃珠，另外八个罪犯也寄给了冯大哥。 |
| 是 → 寄给了张三 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他寄给了张三好几只手套，那六个姐姐也是。<br>Bad: 他寄给了张三好几只手套，那六个姐姐也寄给了张三。 |
| 是 → 寄给了陈大姐 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生寄给了陈大姐六条被子，这位舞者也是。<br>Bad: 王先生寄给了陈大姐六条被子，这位舞者也寄给了陈大姐。 |
| 是 → 送给了刘先生 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷送给了刘先生好几只袜子，那个钢琴家也是。<br>Bad: 胡大爷送给了刘先生好几只袜子，那个钢琴家也送给了刘先生。 |
| 是 → 送给了张夫人 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐送给了张夫人五条裙子，那十位下属也是。<br>Bad: 陈大姐送给了张夫人五条裙子，那十位下属也送给了张夫人。 |
| 是 → 送给了王大娘 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐送给了王大娘十几张桌子，另外五个演员也是。<br>Bad: 徐小姐送给了王大娘十几张桌子，另外五个演员也送给了王大娘。 |
| 是 → 送给了王小姐 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈送给了王小姐九本教材，这八位员工也是。<br>Bad: 周大妈送给了王小姐九本教材，这八位员工也送给了王小姐。 |
| 是 → 送给了胡大爷 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们送给了胡大爷许多把椅子，另外三个小孩也是。<br>Bad: 她们送给了胡大爷许多把椅子，另外三个小孩也送给了胡大爷。 |
| 是 → 送给了陈大姐 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们送给了陈大姐两只手套，那四位父亲也是。<br>Bad: 我们送给了陈大姐两只手套，那四位父亲也送给了陈大姐。 |
| 是 → 递给了吴太太 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们递给了吴太太四只手套，这个小孩也是。<br>Bad: 他们递给了吴太太四只手套，这个小孩也递给了吴太太。 |
| 是 → 递给了宋女士 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明递给了宋女士非常多张桌子，这三位母亲也是。<br>Bad: 小明递给了宋女士非常多张桌子，这三位母亲也递给了宋女士。 |
| 是 → 递给了小王 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷递给了小王七条裙子，那个司机也是。<br>Bad: 胡大爷递给了小王七条裙子，那个司机也递给了小王。 |
| 是 → 递给了徐小姐 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥递给了徐小姐三条裙子，另外六个下属也是。<br>Bad: 杨大哥递给了徐小姐三条裙子，另外六个下属也递给了徐小姐。 |
| 是 → 递给了杨大哥 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四递给了杨大哥五把椅子，这十个领导也是。<br>Bad: 李四递给了杨大哥五把椅子，这十个领导也递给了杨大哥。 |
| 是 → 递给了王小姐 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨递给了王小姐两把椅子，他们的上级也是。<br>Bad: 王姨递给了王小姐两把椅子，他们的上级也递给了王小姐。 |
| 是 → 递给了赵大爷 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐递给了赵大爷九把椅子，这五位领导也是。<br>Bad: 徐小姐递给了赵大爷九把椅子，这五位领导也递给了赵大爷。 |
| 是 → 递给了陈大姐 | ellipsis_double_object | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨递给了陈大姐许多个充电器，那位演员也是。<br>Bad: 王姨递给了陈大姐许多个充电器，那位演员也递给了陈大姐。 |
| multiple edits: 创作 -> 去; 小说 -> 一分钟; 创作 -> 去 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生创作了小说，你也创作了。<br>Bad: 刘先生去了一分钟，你也去了。 |
| multiple edits: 创作 -> 叹息; 漫画 -> 一分钟; 创作 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三创作了漫画，张先生也创作了。<br>Bad: 张三叹息了一分钟，张先生也叹息了。 |
| multiple edits: 创作 -> 品茶; 漫画 -> 一天; 创作 -> 品茶 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他创作了漫画，小明也创作了。<br>Bad: 他品茶了一天，小明也品茶了。 |
| multiple edits: 创作 -> 哭; 小说 -> 一天; 创作 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我创作了小说，你们也创作了。<br>Bad: 我哭了一天，你们也哭了。 |
| multiple edits: 创作 -> 睡觉; 漫画 -> 很久; 创作 -> 睡觉 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生创作了漫画，他们也创作了。<br>Bad: 王先生睡觉了很久，他们也睡觉了。 |
| multiple edits: 创作 -> 站立; 小说 -> 一分钟; 创作 -> 站立 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们创作了小说，她也创作了。<br>Bad: 她们站立了一分钟，她也站立了。 |
| multiple edits: 制作 -> 入睡; 手账 -> 一分钟; 制作 -> 入睡 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们制作了手账，小明也制作了。<br>Bad: 她们入睡了一分钟，小明也入睡了。 |
| multiple edits: 制作 -> 入睡; 电影 -> 一天; 制作 -> 入睡 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我制作了电影，你也制作了。<br>Bad: 我入睡了一天，你也入睡了。 |
| multiple edits: 制作 -> 叹息; 电影 -> 一天; 制作 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生制作了电影，赵大爷也制作了。<br>Bad: 张先生叹息了一天，赵大爷也叹息了。 |
| multiple edits: 制作 -> 呼吸; 动作片 -> 很久; 制作 -> 呼吸 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷制作了动作片，王大娘也制作了。<br>Bad: 胡大爷呼吸了很久，王大娘也呼吸了。 |
| multiple edits: 制作 -> 品茶; 手账 -> 一小时; 制作 -> 品茶 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们制作了手账，她们也制作了。<br>Bad: 我们品茶了一小时，她们也品茶了。 |
| multiple edits: 制作 -> 品茶; 电视剧 -> 一会儿; 制作 -> 品茶 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三制作了电视剧，我们也制作了。<br>Bad: 张三品茶了一会儿，我们也品茶了。 |
| multiple edits: 制作 -> 起飞; 电影 -> 一分钟; 制作 -> 起飞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈制作了电影，小明也制作了。<br>Bad: 郑大妈起飞了一分钟，小明也起飞了。 |
| multiple edits: 制作 -> 跑步; 手账 -> 一分钟; 制作 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你制作了手账，张三也制作了。<br>Bad: 你跑步了一分钟，张三也跑步了。 |
| multiple edits: 制作 -> 过去; 手账 -> 一分钟; 制作 -> 过去 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人制作了手账，你们也制作了。<br>Bad: 张夫人过去了一分钟，你们也过去了。 |
| multiple edits: 制作 -> 闲逛; 电影 -> 一小时; 制作 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他制作了电影，她也制作了。<br>Bad: 他闲逛了一小时，她也闲逛了。 |
| multiple edits: 包扎 -> 坐下; 耳朵 -> 一天; 包扎 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你包扎了耳朵，刘先生也包扎了。<br>Bad: 你坐下了一天，刘先生也坐下了。 |
| multiple edits: 包扎 -> 微笑; 耳朵 -> 一天; 包扎 -> 微笑 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐包扎了耳朵，他也包扎了。<br>Bad: 王小姐微笑了一天，他也微笑了。 |
| multiple edits: 包扎 -> 跳舞; 耳朵 -> 一会儿; 包扎 -> 跳舞 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们包扎了耳朵，你也包扎了。<br>Bad: 她们跳舞了一会儿，你也跳舞了。 |
| multiple edits: 包扎 -> 颤抖; 脚 -> 一天; 包扎 -> 颤抖 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们包扎了脚，你们也包扎了。<br>Bad: 她们颤抖了一天，你们也颤抖了。 |
| multiple edits: 吃 -> 出发; 蛋糕 -> 一会儿; 吃 -> 出发 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们吃了蛋糕，你们也吃了。<br>Bad: 他们出发了一会儿，你们也出发了。 |
| multiple edits: 吃 -> 叹息; 糖 -> 很久; 吃 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐吃了糖，你们也吃了。<br>Bad: 徐小姐叹息了很久，你们也叹息了。 |
| multiple edits: 吃 -> 品茶; 糖 -> 一小时; 吃 -> 品茶 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们吃了糖，王小姐也吃了。<br>Bad: 我们品茶了一小时，王小姐也品茶了。 |
| multiple edits: 吃 -> 走路; 鸡 -> 一分钟; 吃 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士吃了鸡，你们也吃了。<br>Bad: 宋女士走路了一分钟，你们也走路了。 |
| multiple edits: 吹 -> 听课; 双簧 -> 一分钟; 吹 -> 听课 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王吹了双簧，何太太也吹了。<br>Bad: 小王听课了一分钟，何太太也听课了。 |
| multiple edits: 吹 -> 启程; 笛子 -> 很久; 吹 -> 启程 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你吹了笛子，你们也吹了。<br>Bad: 你启程了很久，你们也启程了。 |
| multiple edits: 吹 -> 呼吸; 双簧 -> 一分钟; 吹 -> 呼吸 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她吹了双簧，何太太也吹了。<br>Bad: 她呼吸了一分钟，何太太也呼吸了。 |
| multiple edits: 吹 -> 哭; 双簧 -> 一天; 吹 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐吹了双簧，她们也吹了。<br>Bad: 王小姐哭了一天，她们也哭了。 |
| multiple edits: 吹 -> 哭; 双簧 -> 很久; 吹 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她吹了双簧，徐小姐也吹了。<br>Bad: 她哭了很久，徐小姐也哭了。 |
| multiple edits: 吹 -> 跑步; 笛子 -> 一天; 吹 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐吹了笛子，她们也吹了。<br>Bad: 陈大姐跑步了一天，她们也跑步了。 |
| multiple edits: 吹 -> 闲逛; 笛子 -> 一分钟; 吹 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我吹了笛子，张先生也吹了。<br>Bad: 我闲逛了一分钟，张先生也闲逛了。 |
| multiple edits: 唱 -> 偷听; 戏曲 -> 很久; 唱 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生唱了戏曲，冯大哥也唱了。<br>Bad: 张先生偷听了很久，冯大哥也偷听了。 |
| multiple edits: 唱 -> 打架; 小调 -> 一天; 唱 -> 打架 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三唱了小调，我们也唱了。<br>Bad: 张三打架了一天，我们也打架了。 |
| multiple edits: 唱 -> 打架; 美声 -> 一分钟; 唱 -> 打架 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷唱了美声，我们也唱了。<br>Bad: 赵大爷打架了一分钟，我们也打架了。 |
| multiple edits: 唱 -> 溜走; bad inserts 一; 调 -> 时; 唱 -> 溜走 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷唱了小调，王先生也唱了。<br>Bad: 胡大爷溜走了一小时，王先生也溜走了。 |
| multiple edits: 唱 -> 爬行; 歌 -> 一会儿; 唱 -> 爬行 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐唱了歌，她也唱了。<br>Bad: 陈大姐爬行了一会儿，她也爬行了。 |
| multiple edits: 唱 -> 站立; 歌 -> 一天; 唱 -> 站立 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三唱了歌，宋女士也唱了。<br>Bad: 张三站立了一天，宋女士也站立了。 |
| multiple edits: 唱 -> 笑; 戏曲 -> 一会儿; 唱 -> 笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明唱了戏曲，他们也唱了。<br>Bad: 小明笑了一会儿，他们也笑了。 |
| multiple edits: 唱 -> 闲逛; 小调 -> 一分钟; 唱 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我唱了小调，张三也唱了。<br>Bad: 我闲逛了一分钟，张三也闲逛了。 |
| multiple edits: 唱 -> 闲逛; 小调 -> 一天; 唱 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我唱了小调，徐小姐也唱了。<br>Bad: 我闲逛了一天，徐小姐也闲逛了。 |
| multiple edits: 喝 -> 偷听; 橙汁 -> 一会儿; 喝 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生喝了橙汁，王大娘也喝了。<br>Bad: 王先生偷听了一会儿，王大娘也偷听了。 |
| multiple edits: 喝 -> 听课; 红酒 -> 一会儿; 喝 -> 听课 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们喝了红酒，你们也喝了。<br>Bad: 他们听课了一会儿，你们也听课了。 |
| multiple edits: 喝 -> 呼吸; 冰红茶 -> 一天; 喝 -> 呼吸 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈喝了冰红茶，你也喝了。<br>Bad: 周大妈呼吸了一天，你也呼吸了。 |
| multiple edits: 喝 -> 呼吸; 红茶 -> 一分钟; 喝 -> 呼吸 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷喝了红茶，她也喝了。<br>Bad: 赵大爷呼吸了一分钟，她也呼吸了。 |
| multiple edits: 喝 -> 打架; 啤酒 -> 一会儿; 喝 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他喝了啤酒，王五也喝了。<br>Bad: 他打架了一会儿，王五也打架了。 |
| multiple edits: 喝 -> 起飞; 橙汁 -> 一分钟; 喝 -> 起飞 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他喝了橙汁，我也喝了。<br>Bad: 他起飞了一分钟，我也起飞了。 |
| multiple edits: 喝 -> 跳舞; 白酒 -> 一分钟; 喝 -> 跳舞 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们喝了白酒，我也喝了。<br>Bad: 她们跳舞了一分钟，我也跳舞了。 |
| multiple edits: 喝 -> 过去; 白酒 -> 一小时; 喝 -> 过去 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太喝了白酒，王小姐也喝了。<br>Bad: 李太太过去了一小时，王小姐也过去了。 |
| multiple edits: 喝 -> 闲逛; 啤酒 -> 一会儿; 喝 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我喝了啤酒，胡大爷也喝了。<br>Bad: 我闲逛了一会儿，胡大爷也闲逛了。 |
| multiple edits: 喝 -> 闲逛; 啤酒 -> 一分钟; 喝 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生喝了啤酒，胡大爷也喝了。<br>Bad: 王先生闲逛了一分钟，胡大爷也闲逛了。 |
| multiple edits: 喝 -> 颤抖; 啤酒 -> 很久; 喝 -> 颤抖 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生喝了啤酒，我们也喝了。<br>Bad: 张先生颤抖了很久，我们也颤抖了。 |
| multiple edits: 屠宰 -> 呼吸; 牛 -> 很久; 屠宰 -> 呼吸 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们屠宰了牛，你们也屠宰了。<br>Bad: 我们呼吸了很久，你们也呼吸了。 |
| multiple edits: 屠宰 -> 坐下; 牛 -> 一天; 屠宰 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们屠宰了牛，她也屠宰了。<br>Bad: 我们坐下了一天，她也坐下了。 |
| multiple edits: 屠宰 -> 坐下; 牛 -> 一小时; 屠宰 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太屠宰了牛，冯大哥也屠宰了。<br>Bad: 吴太太坐下了一小时，冯大哥也坐下了。 |
| multiple edits: 屠宰 -> 来; 牛 -> 一分钟; 屠宰 -> 来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈屠宰了牛，我也屠宰了。<br>Bad: 周大妈来了一分钟，我也来了。 |
| multiple edits: 屠宰 -> 过去; 牛 -> 一天; 屠宰 -> 过去 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们屠宰了牛，她也屠宰了。<br>Bad: 她们过去了一天，她也过去了。 |
| multiple edits: 屠宰 -> 闲逛; 牛 -> 很久; 屠宰 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生屠宰了牛，她也屠宰了。<br>Bad: 李先生闲逛了很久，她也闲逛了。 |
| multiple edits: 开 -> 健身; 轮船 -> 一会儿; 开 -> 健身 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我开了轮船，王小姐也开了。<br>Bad: 我健身了一会儿，王小姐也健身了。 |
| multiple edits: 开 -> 出发; 飞机 -> 一会儿; 开 -> 出发 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们开了飞机，你也开了。<br>Bad: 他们出发了一会儿，你也出发了。 |
| multiple edits: 开 -> 唱歌; 飞机 -> 很久; 开 -> 唱歌 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五开了飞机，李太太也开了。<br>Bad: 王五唱歌了很久，李太太也唱歌了。 |
| multiple edits: 开 -> 打架; 火车 -> 一会儿; 开 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我开了火车，他们也开了。<br>Bad: 我打架了一会儿，他们也打架了。 |
| multiple edits: 开 -> 打架; 轮船 -> 一小时; 开 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明开了轮船，我也开了。<br>Bad: 小明打架了一小时，我也打架了。 |
| multiple edits: 开 -> 看戏; 轮船 -> 一天; 开 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四开了轮船，王小姐也开了。<br>Bad: 李四看戏了一天，王小姐也看戏了。 |
| multiple edits: 开 -> 笑; 卡车 -> 一小时; 开 -> 笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈开了卡车，我也开了。<br>Bad: 郑大妈笑了一小时，我也笑了。 |
| multiple edits: 开 -> 走; 飞机 -> 一天; 开 -> 走 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们开了飞机，我也开了。<br>Bad: 你们走了一天，我也走了。 |
| multiple edits: 开 -> 走路; 火车 -> 一天; 开 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们开了火车，我们也开了。<br>Bad: 你们走路了一天，我们也走路了。 |
| multiple edits: 开 -> 起飞; 火车 -> 一会儿; 开 -> 起飞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶开了火车，我也开了。<br>Bad: 张婶起飞了一会儿，我也起飞了。 |
| multiple edits: 开 -> 跳舞; 火车 -> 一会儿; 开 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太开了火车，他也开了。<br>Bad: 李太太跳舞了一会儿，他也跳舞了。 |
| multiple edits: 开 -> 躺下; 卡车 -> 一会儿; 开 -> 躺下 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们开了卡车，我们也开了。<br>Bad: 你们躺下了一会儿，我们也躺下了。 |
| multiple edits: 弹 -> 叹息; 玻璃珠 -> 一分钟; 弹 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们弹了玻璃珠，你也弹了。<br>Bad: 他们叹息了一分钟，你也叹息了。 |
| multiple edits: 弹 -> 启程; 玻璃珠 -> 一会儿; 弹 -> 启程 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你弹了玻璃珠，他也弹了。<br>Bad: 你启程了一会儿，他也启程了。 |
| multiple edits: 弹 -> 游泳; 玻璃珠 -> 一分钟; 弹 -> 游泳 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们弹了玻璃珠，张先生也弹了。<br>Bad: 你们游泳了一分钟，张先生也游泳了。 |
| multiple edits: 弹 -> 看戏; 玻璃珠 -> 一天; 弹 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生弹了玻璃珠，你们也弹了。<br>Bad: 王先生看戏了一天，你们也看戏了。 |
| multiple edits: 弹 -> 睡觉; 玻璃珠 -> 很久; 弹 -> 睡觉 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们弹了玻璃珠，何太太也弹了。<br>Bad: 我们睡觉了很久，何太太也睡觉了。 |
| multiple edits: 弹 -> 笑; 玻璃珠 -> 很久; 弹 -> 笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太弹了玻璃珠，张三也弹了。<br>Bad: 李太太笑了很久，张三也笑了。 |
| multiple edits: 弹 -> 走路; 玻璃珠 -> 很久; 弹 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶弹了玻璃珠，徐小姐也弹了。<br>Bad: 张婶走路了很久，徐小姐也走路了。 |
| multiple edits: 打断 -> 坐下; 手 -> 一天; 打断 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太打断了手，你也打断了。<br>Bad: 吴太太坐下了一天，你也坐下了。 |
| multiple edits: 打断 -> 起飞; 腿 -> 一小时; 打断 -> 起飞 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你打断了腿，我们也打断了。<br>Bad: 你起飞了一小时，我们也起飞了。 |
| multiple edits: 打断 -> 躺下; 脚 -> 很久; 打断 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们打断了脚，陈大姐也打断了。<br>Bad: 你们躺下了很久，陈大姐也躺下了。 |
| multiple edits: 打断 -> 运动; 鼻子 -> 很久; 打断 -> 运动 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们打断了鼻子，小明也打断了。<br>Bad: 你们运动了很久，小明也运动了。 |
| multiple edits: 拉 -> 停下; 小提琴 -> 一会儿; 拉 -> 停下 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们拉了小提琴，张三也拉了。<br>Bad: 他们停下了一会儿，张三也停下了。 |
| multiple edits: 拉 -> 打架; 小提琴 -> 一会儿; 拉 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生拉了小提琴，刘先生也拉了。<br>Bad: 王先生打架了一会儿，刘先生也打架了。 |
| multiple edits: 拉 -> 玩耍; 小提琴 -> 一分钟; 拉 -> 玩耍 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥拉了小提琴，胡大爷也拉了。<br>Bad: 杨大哥玩耍了一分钟，胡大爷也玩耍了。 |
| multiple edits: 拉 -> 走路; bad inserts 一; 提琴 -> 时; 拉 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们拉了小提琴，她们也拉了。<br>Bad: 你们走路了一小时，她们也走路了。 |
| multiple edits: 拉 -> 跑步; 大提琴 -> 一分钟; 拉 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你拉了大提琴，王大娘也拉了。<br>Bad: 你跑步了一分钟，王大娘也跑步了。 |
| multiple edits: 拉 -> 跑步; 小提琴 -> 一天; 拉 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我拉了小提琴，你也拉了。<br>Bad: 我跑步了一天，你也跑步了。 |
| multiple edits: 拍摄 -> 出发; 电影 -> 一分钟; 拍摄 -> 出发 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷拍摄了电影，她们也拍摄了。<br>Bad: 赵大爷出发了一分钟，她们也出发了。 |
| multiple edits: 拍摄 -> 哭; 电影 -> 一分钟; 拍摄 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥拍摄了电影，她也拍摄了。<br>Bad: 杨大哥哭了一分钟，她也哭了。 |
| multiple edits: 拍摄 -> 微笑; 电影 -> 一分钟; 拍摄 -> 微笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你拍摄了电影，徐小姐也拍摄了。<br>Bad: 你微笑了一分钟，徐小姐也微笑了。 |
| multiple edits: 拍摄 -> 打架; 电影 -> 一小时; 拍摄 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他拍摄了电影，王大娘也拍摄了。<br>Bad: 他打架了一小时，王大娘也打架了。 |
| multiple edits: 拍摄 -> 溜走; 电影 -> 很久; 拍摄 -> 溜走 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太拍摄了电影，她也拍摄了。<br>Bad: 吴太太溜走了很久，她也溜走了。 |
| multiple edits: 拍摄 -> 爬行; 电影 -> 一分钟; 拍摄 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明拍摄了电影，你们也拍摄了。<br>Bad: 小明爬行了一分钟，你们也爬行了。 |
| multiple edits: 拍摄 -> 爬行; 电影 -> 很久; 拍摄 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他拍摄了电影，赵大爷也拍摄了。<br>Bad: 他爬行了很久，赵大爷也爬行了。 |
| multiple edits: 拍摄 -> 看戏; 动作片 -> 一天; 拍摄 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三拍摄了动作片，小明也拍摄了。<br>Bad: 张三看戏了一天，小明也看戏了。 |
| multiple edits: 拍摄 -> 看戏; 电影 -> 一分钟; 拍摄 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太拍摄了电影，你们也拍摄了。<br>Bad: 何太太看戏了一分钟，你们也看戏了。 |
| multiple edits: 拍摄 -> 跳舞; 电影 -> 一会儿; 拍摄 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明拍摄了电影，徐小姐也拍摄了。<br>Bad: 小明跳舞了一会儿，徐小姐也跳舞了。 |
| multiple edits: 拍摄 -> 运动; 电影 -> 一会儿; 拍摄 -> 运动 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘拍摄了电影，李太太也拍摄了。<br>Bad: 王大娘运动了一会儿，李太太也运动了。 |
| multiple edits: 拍摄 -> 颤抖; 动作片 -> 一天; 拍摄 -> 颤抖 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们拍摄了动作片，张先生也拍摄了。<br>Bad: 她们颤抖了一天，张先生也颤抖了。 |
| multiple edits: 拍摄 -> 颤抖; 电影 -> 一会儿; 拍摄 -> 颤抖 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈拍摄了电影，王先生也拍摄了。<br>Bad: 郑大妈颤抖了一会儿，王先生也颤抖了。 |
| multiple edits: 捕捉 -> 健身; 鱼 -> 一会儿; 捕捉 -> 健身 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥捕捉了鱼，王大娘也捕捉了。<br>Bad: 杨大哥健身了一会儿，王大娘也健身了。 |
| multiple edits: 捕捉 -> 偷听; 大象 -> 一小时; 捕捉 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他捕捉了大象，徐小姐也捕捉了。<br>Bad: 他偷听了一小时，徐小姐也偷听了。 |
| multiple edits: 捕捉 -> 叹息; 鱼 -> 很久; 捕捉 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们捕捉了鱼，张三也捕捉了。<br>Bad: 他们叹息了很久，张三也叹息了。 |
| multiple edits: 捕捉 -> 哭; 蛇 -> 很久; 捕捉 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生捕捉了蛇，张夫人也捕捉了。<br>Bad: 刘先生哭了很久，张夫人也哭了。 |
| multiple edits: 捕捉 -> 玩耍; 鸭 -> 很久; 捕捉 -> 玩耍 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们捕捉了鸭，你们也捕捉了。<br>Bad: 我们玩耍了很久，你们也玩耍了。 |
| multiple edits: 捕捉 -> 笑; 大象 -> 一小时; 捕捉 -> 笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐捕捉了大象，李太太也捕捉了。<br>Bad: 王小姐笑了一小时，李太太也笑了。 |
| multiple edits: 捕捉 -> 起飞; 鸭 -> 很久; 捕捉 -> 起飞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生捕捉了鸭，李太太也捕捉了。<br>Bad: 刘先生起飞了很久，李太太也起飞了。 |
| multiple edits: 捕捉 -> 颤抖; 鸭 -> 一天; 捕捉 -> 颤抖 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人捕捉了鸭，他也捕捉了。<br>Bad: 张夫人颤抖了一天，他也颤抖了。 |
| multiple edits: 检查 -> 偷听; 胃 -> 一分钟; 检查 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他检查了胃，他们也检查了。<br>Bad: 他偷听了一分钟，他们也偷听了。 |
| multiple edits: 检查 -> 玩耍; 脚 -> 一会儿; 检查 -> 玩耍 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐检查了脚，张三也检查了。<br>Bad: 陈大姐玩耍了一会儿，张三也玩耍了。 |
| multiple edits: 检查 -> 看戏; 胃 -> 一会儿; 检查 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她检查了胃，吴太太也检查了。<br>Bad: 她看戏了一会儿，吴太太也看戏了。 |
| multiple edits: 检查 -> 跑步; 腿 -> 一分钟; 检查 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生检查了腿，你也检查了。<br>Bad: 张先生跑步了一分钟，你也跑步了。 |
| multiple edits: 清洗 -> 听课; 杯子 -> 一会儿; 清洗 -> 听课 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她清洗了杯子，何太太也清洗了。<br>Bad: 她听课了一会儿，何太太也听课了。 |
| multiple edits: 清洗 -> 唱歌; 杯子 -> 很久; 清洗 -> 唱歌 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太清洗了杯子，李四也清洗了。<br>Bad: 吴太太唱歌了很久，李四也唱歌了。 |
| multiple edits: 清洗 -> 打架; 杯子 -> 很久; 清洗 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们清洗了杯子，张夫人也清洗了。<br>Bad: 你们打架了很久，张夫人也打架了。 |
| multiple edits: 清洗 -> 游泳; 杯子 -> 一分钟; 清洗 -> 游泳 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们清洗了杯子，我们也清洗了。<br>Bad: 他们游泳了一分钟，我们也游泳了。 |
| multiple edits: 清洗 -> 过去; 杯子 -> 很久; 清洗 -> 过去 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明清洗了杯子，张夫人也清洗了。<br>Bad: 小明过去了很久，张夫人也过去了。 |
| multiple edits: 清洗 -> 过来; 杯子 -> 一分钟; 清洗 -> 过来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈清洗了杯子，他也清洗了。<br>Bad: 郑大妈过来了一分钟，他也过来了。 |
| multiple edits: 清洗 -> 闲逛; 杯子 -> 一小时; 清洗 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们清洗了杯子，张先生也清洗了。<br>Bad: 他们闲逛了一小时，张先生也闲逛了。 |
| multiple edits: 清蒸 -> 启程; 鱼 -> 一天; 清蒸 -> 启程 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈清蒸了鱼，我们也清蒸了。<br>Bad: 周大妈启程了一天，我们也启程了。 |
| multiple edits: 清蒸 -> 呼吸; 鸭 -> 一会儿; 清蒸 -> 呼吸 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们清蒸了鸭，我们也清蒸了。<br>Bad: 她们呼吸了一会儿，我们也呼吸了。 |
| multiple edits: 清蒸 -> 走路; 鸭 -> 一小时; 清蒸 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们清蒸了鸭，王先生也清蒸了。<br>Bad: 她们走路了一小时，王先生也走路了。 |
| multiple edits: 清蒸 -> 跑步; 鸭 -> 很久; 清蒸 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们清蒸了鸭，他也清蒸了。<br>Bad: 你们跑步了很久，他也跑步了。 |
| multiple edits: 清蒸 -> 躺下; 鱼 -> 一天; 清蒸 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我清蒸了鱼，她们也清蒸了。<br>Bad: 我躺下了一天，她们也躺下了。 |
| multiple edits: 清蒸 -> 运动; 鸭 -> 一小时; 清蒸 -> 运动 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈清蒸了鸭，张夫人也清蒸了。<br>Bad: 周大妈运动了一小时，张夫人也运动了。 |
| multiple edits: 演奏 -> 打架; 奏鸣曲 -> 一小时; 演奏 -> 打架 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她演奏了奏鸣曲，李四也演奏了。<br>Bad: 她打架了一小时，李四也打架了。 |
| multiple edits: 演奏 -> 走路; 歌曲 -> 一会儿; 演奏 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈演奏了歌曲，你也演奏了。<br>Bad: 周大妈走路了一会儿，你也走路了。 |
| multiple edits: 演奏 -> 跳舞; 奏鸣曲 -> 一会儿; 演奏 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们演奏了奏鸣曲，王先生也演奏了。<br>Bad: 他们跳舞了一会儿，王先生也跳舞了。 |
| multiple edits: 炖 -> 叹息; 鱼 -> 一天; 炖 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们炖了鱼，郑大妈也炖了。<br>Bad: 你们叹息了一天，郑大妈也叹息了。 |
| multiple edits: 炖 -> 哭; 鸭 -> 很久; 炖 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生炖了鸭，赵大爷也炖了。<br>Bad: 刘先生哭了很久，赵大爷也哭了。 |
| multiple edits: 炖 -> 玩耍; 鸡 -> 一天; 炖 -> 玩耍 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王炖了鸡，她也炖了。<br>Bad: 小王玩耍了一天，她也玩耍了。 |
| multiple edits: 炖 -> 笑; 鱼 -> 一小时; 炖 -> 笑 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生炖了鱼，我们也炖了。<br>Bad: 王先生笑了一小时，我们也笑了。 |
| multiple edits: 炖 -> 过去; 鱼 -> 一小时; 炖 -> 过去 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷炖了鱼，她们也炖了。<br>Bad: 胡大爷过去了一小时，她们也过去了。 |
| multiple edits: 烧 -> 去; 鱼 -> 一天; 烧 -> 去 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘烧了鱼，我们也烧了。<br>Bad: 王大娘去了一天，我们也去了。 |
| multiple edits: 烧 -> 哭; 鱼 -> 一会儿; 烧 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你烧了鱼，他也烧了。<br>Bad: 你哭了一会儿，他也哭了。 |
| multiple edits: 烧 -> 哭; 鸭 -> 一小时; 烧 -> 哭 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们烧了鸭，小王也烧了。<br>Bad: 她们哭了一小时，小王也哭了。 |
| multiple edits: 烧 -> 躺下; 鱼 -> 一天; 烧 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生烧了鱼，他们也烧了。<br>Bad: 王先生躺下了一天，他们也躺下了。 |
| multiple edits: 烧 -> 闲逛; 鸡 -> 很久; 烧 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你烧了鸡，她们也烧了。<br>Bad: 你闲逛了很久，她们也闲逛了。 |
| multiple edits: 煮 -> 听课; 鸭 -> 很久; 煮 -> 听课 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥煮了鸭，她们也煮了。<br>Bad: 杨大哥听课了很久，她们也听课了。 |
| multiple edits: 煮 -> 打架; 鸡 -> 一小时; 煮 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈煮了鸡，你也煮了。<br>Bad: 周大妈打架了一小时，你也打架了。 |
| multiple edits: 煮 -> 笑; 鸡 -> 很久; 煮 -> 笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他煮了鸡，刘先生也煮了。<br>Bad: 他笑了很久，刘先生也笑了。 |
| multiple edits: 煮 -> 跳舞; 鸡 -> 一会儿; 煮 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生煮了鸡，她们也煮了。<br>Bad: 刘先生跳舞了一会儿，她们也跳舞了。 |
| multiple edits: 煮 -> 跳舞; 鸭 -> 一会儿; 煮 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你煮了鸭，他也煮了。<br>Bad: 你跳舞了一会儿，他也跳舞了。 |
| multiple edits: 煮 -> 运动; 鸭 -> 一小时; 煮 -> 运动 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐煮了鸭，我们也煮了。<br>Bad: 徐小姐运动了一小时，我们也运动了。 |
| multiple edits: 爆炒 -> 启程; 鱼 -> 一天; 爆炒 -> 启程 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们爆炒了鱼，张夫人也爆炒了。<br>Bad: 你们启程了一天，张夫人也启程了。 |
| multiple edits: 爆炒 -> 品茶; 鸡 -> 一分钟; 爆炒 -> 品茶 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈爆炒了鸡，王先生也爆炒了。<br>Bad: 郑大妈品茶了一分钟，王先生也品茶了。 |
| multiple edits: 爆炒 -> 哭; 鸡 -> 一小时; 爆炒 -> 哭 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们爆炒了鸡，我也爆炒了。<br>Bad: 她们哭了一小时，我也哭了。 |
| multiple edits: 爆炒 -> 游泳; 鸡 -> 一天; 爆炒 -> 游泳 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们爆炒了鸡，你也爆炒了。<br>Bad: 他们游泳了一天，你也游泳了。 |
| multiple edits: 爆炒 -> 玩耍; 鱼 -> 很久; 爆炒 -> 玩耍 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶爆炒了鱼，我们也爆炒了。<br>Bad: 张婶玩耍了很久，我们也玩耍了。 |
| multiple edits: 爆炒 -> 看戏; 鸭 -> 一天; 爆炒 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们爆炒了鸭，我也爆炒了。<br>Bad: 他们看戏了一天，我也看戏了。 |
| multiple edits: 爆炒 -> 看戏; 鸭 -> 很久; 爆炒 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘爆炒了鸭，我也爆炒了。<br>Bad: 王大娘看戏了很久，我也看戏了。 |
| multiple edits: 爆炒 -> 走; 鸡 -> 一会儿; 爆炒 -> 走 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷爆炒了鸡，小王也爆炒了。<br>Bad: 赵大爷走了一会儿，小王也走了。 |
| multiple edits: 爆炒 -> 跑步; 鸭 -> 一天; 爆炒 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他爆炒了鸭，何太太也爆炒了。<br>Bad: 他跑步了一天，何太太也跑步了。 |
| multiple edits: 爆炒 -> 闲逛; 鸡 -> 很久; 爆炒 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷爆炒了鸡，你也爆炒了。<br>Bad: 胡大爷闲逛了很久，你也闲逛了。 |
| multiple edits: 盖 -> 偷听; 被子 -> 一小时; 盖 -> 偷听 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶盖了被子，杨大哥也盖了。<br>Bad: 张婶偷听了一小时，杨大哥也偷听了。 |
| multiple edits: 盖 -> 唱歌; 被子 -> 一会儿; 盖 -> 唱歌 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他盖了被子，胡大爷也盖了。<br>Bad: 他唱歌了一会儿，胡大爷也唱歌了。 |
| multiple edits: 盖 -> 坐下; 被子 -> 一分钟; 盖 -> 坐下 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生盖了被子，他们也盖了。<br>Bad: 刘先生坐下了一分钟，他们也坐下了。 |
| multiple edits: 盖 -> 爬行; 被子 -> 一小时; 盖 -> 爬行 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她盖了被子，张先生也盖了。<br>Bad: 她爬行了一小时，张先生也爬行了。 |
| multiple edits: 盖 -> 玩耍; 被子 -> 一天; 盖 -> 玩耍 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐盖了被子，王姨也盖了。<br>Bad: 陈大姐玩耍了一天，王姨也玩耍了。 |
| multiple edits: 盖 -> 看戏; 被子 -> 很久; 盖 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我盖了被子，陈大姐也盖了。<br>Bad: 我看戏了很久，陈大姐也看戏了。 |
| multiple edits: 盖 -> 跑步; 被子 -> 很久; 盖 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四盖了被子，他们也盖了。<br>Bad: 李四跑步了很久，他们也跑步了。 |
| multiple edits: 盖 -> 颤抖; 被子 -> 一天; 盖 -> 颤抖 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你盖了被子，张三也盖了。<br>Bad: 你颤抖了一天，张三也颤抖了。 |
| multiple edits: 看 -> 睡觉; 录像带 -> 一天; 看 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生看了录像带，她们也看了。<br>Bad: 王先生睡觉了一天，她们也睡觉了。 |
| multiple edits: 看 -> 躺下; 书 -> 很久; 看 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我看了书，你们也看了。<br>Bad: 我躺下了很久，你们也躺下了。 |
| multiple edits: 看 -> 躺下; 录像带 -> 一会儿; 看 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她看了录像带，刘先生也看了。<br>Bad: 她躺下了一会儿，刘先生也躺下了。 |
| multiple edits: 看 -> 闲逛; 书 -> 一天; 看 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥看了书，何太太也看了。<br>Bad: 冯大哥闲逛了一天，何太太也闲逛了。 |
| multiple edits: 看 -> 闲逛; 书 -> 一小时; 看 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们看了书，张婶也看了。<br>Bad: 她们闲逛了一小时，张婶也闲逛了。 |
| multiple edits: 观看 -> 停下; 电视剧 -> 一天; 观看 -> 停下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷观看了电视剧，你也观看了。<br>Bad: 胡大爷停下了一天，你也停下了。 |
| multiple edits: 观看 -> 听课; 电视剧 -> 一小时; 观看 -> 听课 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他观看了电视剧，杨大哥也观看了。<br>Bad: 他听课了一小时，杨大哥也听课了。 |
| multiple edits: 观看 -> 品茶; 电视剧 -> 一小时; 观看 -> 品茶 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘观看了电视剧，他也观看了。<br>Bad: 王大娘品茶了一小时，他也品茶了。 |
| multiple edits: 观看 -> 溜走; 动画片 -> 一会儿; 观看 -> 溜走 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他观看了动画片，她也观看了。<br>Bad: 他溜走了一会儿，她也溜走了。 |
| multiple edits: 观看 -> 躺下; 电影 -> 一会儿; 观看 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太观看了电影，她们也观看了。<br>Bad: 何太太躺下了一会儿，她们也躺下了。 |
| multiple edits: 观看 -> 躺下; 电影 -> 一小时; 观看 -> 躺下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷观看了电影，张夫人也观看了。<br>Bad: 胡大爷躺下了一小时，张夫人也躺下了。 |
| multiple edits: 观看 -> 过来; 动作片 -> 一天; 观看 -> 过来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生观看了动作片，他们也观看了。<br>Bad: 刘先生过来了一天，他们也过来了。 |
| multiple edits: 观看 -> 过来; 电影 -> 一天; 观看 -> 过来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们观看了电影，他们也观看了。<br>Bad: 她们过来了一天，他们也过来了。 |
| multiple edits: 观看 -> 闲逛; 电影 -> 一小时; 观看 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生观看了电影，王五也观看了。<br>Bad: 王先生闲逛了一小时，王五也闲逛了。 |
| multiple edits: 跨越 -> 叹息; 沙漠 -> 很久; 跨越 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们跨越了沙漠，刘先生也跨越了。<br>Bad: 他们叹息了很久，刘先生也叹息了。 |
| multiple edits: 跨越 -> 呼吸; 海洋 -> 一天; 跨越 -> 呼吸 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我跨越了海洋，胡大爷也跨越了。<br>Bad: 我呼吸了一天，胡大爷也呼吸了。 |
| multiple edits: 跨越 -> 坐下; 海洋 -> 一小时; 跨越 -> 坐下 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们跨越了海洋，她也跨越了。<br>Bad: 他们坐下了一小时，她也坐下了。 |
| multiple edits: 跨越 -> 微笑; 海洋 -> 一天; 跨越 -> 微笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四跨越了海洋，她们也跨越了。<br>Bad: 李四微笑了一天，她们也微笑了。 |
| multiple edits: 跨越 -> 跑步; 海洋 -> 一分钟; 跨越 -> 跑步 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们跨越了海洋，你也跨越了。<br>Bad: 我们跑步了一分钟，你也跑步了。 |
| multiple edits: 跨越 -> 过来; 沙漠 -> 一小时; 跨越 -> 过来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生跨越了沙漠，他们也跨越了。<br>Bad: 李先生过来了一小时，他们也过来了。 |
| multiple edits: 跨越 -> 闲逛; 海洋 -> 一小时; 跨越 -> 闲逛 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他跨越了海洋，小王也跨越了。<br>Bad: 他闲逛了一小时，小王也闲逛了。 |
| multiple edits: 预习 -> 启程; 教材 -> 一会儿; 预习 -> 启程 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我预习了教材，你也预习了。<br>Bad: 我启程了一会儿，你也启程了。 |
| multiple edits: 预习 -> 品茶; 教材 -> 一分钟; 预习 -> 品茶 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生预习了教材，你也预习了。<br>Bad: 张先生品茶了一分钟，你也品茶了。 |
| multiple edits: 预习 -> 坐下; 教材 -> 一小时; 预习 -> 坐下 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷预习了教材，他也预习了。<br>Bad: 赵大爷坐下了一小时，他也坐下了。 |
| multiple edits: 预习 -> 溜走; 教材 -> 一小时; 预习 -> 溜走 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她预习了教材，周大妈也预习了。<br>Bad: 她溜走了一小时，周大妈也溜走了。 |
| multiple edits: 预习 -> 跳舞; 教材 -> 一分钟; 预习 -> 跳舞 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生预习了教材，吴太太也预习了。<br>Bad: 张先生跳舞了一分钟，吴太太也跳舞了。 |
| multiple edits: 预习 -> 运动; 教材 -> 一天; 预习 -> 运动 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐预习了教材，徐小姐也预习了。<br>Bad: 陈大姐运动了一天，徐小姐也运动了。 |
| multiple edits: 领养 -> 健身; 小狗 -> 一天; 领养 -> 健身 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你领养了小狗，王姨也领养了。<br>Bad: 你健身了一天，王姨也健身了。 |
| multiple edits: 领养 -> 听课; 小猫 -> 很久; 领养 -> 听课 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们领养了小猫，我也领养了。<br>Bad: 我们听课了很久，我也听课了。 |
| multiple edits: 领养 -> 哭; 小狗 -> 一分钟; 领养 -> 哭 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他领养了小狗，赵大爷也领养了。<br>Bad: 他哭了一分钟，赵大爷也哭了。 |
| multiple edits: 领养 -> 来; 小狗 -> 一天; 领养 -> 来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她领养了小狗，王姨也领养了。<br>Bad: 她来了一天，王姨也来了。 |
| multiple edits: 领养 -> 溜走; 小猫 -> 一会儿; 领养 -> 溜走 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生领养了小猫，她们也领养了。<br>Bad: 王先生溜走了一会儿，她们也溜走了。 |
| multiple edits: 领养 -> 睡觉; 小狗 -> 很久; 领养 -> 睡觉 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶领养了小狗，王五也领养了。<br>Bad: 张婶睡觉了很久，王五也睡觉了。 |
| multiple edits: 领养 -> 站立; 小狗 -> 一会儿; 领养 -> 站立 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷领养了小狗，你也领养了。<br>Bad: 胡大爷站立了一会儿，你也站立了。 |
| multiple edits: 领养 -> 站立; 小猫 -> 一分钟; 领养 -> 站立 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘领养了小猫，陈大姐也领养了。<br>Bad: 王大娘站立了一分钟，陈大姐也站立了。 |
| multiple edits: 领养 -> 走; 小猫 -> 一会儿; 领养 -> 走 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王领养了小猫，胡大爷也领养了。<br>Bad: 小王走了一会儿，胡大爷也走了。 |
| multiple edits: 领养 -> 起飞; bad inserts 一; 猫 -> 时; 领养 -> 起飞 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨领养了小猫，李四也领养了。<br>Bad: 王姨起飞了一小时，李四也起飞了。 |
| multiple edits: 领养 -> 过来; 小猫 -> 一分钟; 领养 -> 过来 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他领养了小猫，张婶也领养了。<br>Bad: 他过来了一分钟，张婶也过来了。 |
| multiple edits: 驾驶 -> 出发; 卡车 -> 一天; 驾驶 -> 出发 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷驾驶了卡车，他们也驾驶了。<br>Bad: 赵大爷出发了一天，他们也出发了。 |
| multiple edits: 驾驶 -> 唱歌; 卡车 -> 一会儿; 驾驶 -> 唱歌 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生驾驶了卡车，你们也驾驶了。<br>Bad: 李先生唱歌了一会儿，你们也唱歌了。 |
| multiple edits: 驾驶 -> 坐下; 卡车 -> 一天; 驾驶 -> 坐下 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们驾驶了卡车，我们也驾驶了。<br>Bad: 你们坐下了一天，我们也坐下了。 |
| multiple edits: 驾驶 -> 打架; 火车 -> 一天; 驾驶 -> 打架 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我驾驶了火车，你们也驾驶了。<br>Bad: 我打架了一天，你们也打架了。 |
| multiple edits: 驾驶 -> 看戏; 飞机 -> 一天; 驾驶 -> 看戏 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王驾驶了飞机，陈大姐也驾驶了。<br>Bad: 小王看戏了一天，陈大姐也看戏了。 |
| multiple edits: 驾驶 -> 走路; 火车 -> 一小时; 驾驶 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我驾驶了火车，张三也驾驶了。<br>Bad: 我走路了一小时，张三也走路了。 |
| multiple edits: 驾驶 -> 走路; 货车 -> 一小时; 驾驶 -> 走路 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太驾驶了货车，陈大姐也驾驶了。<br>Bad: 李太太走路了一小时，陈大姐也走路了。 |
| multiple edits: 驾驶 -> 跳舞; 货车 -> 一分钟; 驾驶 -> 跳舞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四驾驶了货车，我也驾驶了。<br>Bad: 李四跳舞了一分钟，我也跳舞了。 |
| multiple edits: 驾驶 -> 运动; 火车 -> 一分钟; 驾驶 -> 运动 | ellipsis_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生驾驶了火车，我也驾驶了。<br>Bad: 张先生运动了一分钟，我也运动了。 |
| multiple edits: 驾驶 -> 颤抖; 飞机 -> 一小时; 驾驶 -> 颤抖 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈驾驶了飞机，你也驾驶了。<br>Bad: 郑大妈颤抖了一小时，你也颤抖了。 |
| multiple edits: 麻醉 -> 出发; 大象 -> 很久; 麻醉 -> 出发 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷麻醉了大象，她们也麻醉了。<br>Bad: 赵大爷出发了很久，她们也出发了。 |
| multiple edits: 麻醉 -> 叹息; 大象 -> 一天; 麻醉 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥麻醉了大象，王五也麻醉了。<br>Bad: 杨大哥叹息了一天，王五也叹息了。 |
| multiple edits: 麻醉 -> 叹息; 老虎 -> 一分钟; 麻醉 -> 叹息 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们麻醉了老虎，郑大妈也麻醉了。<br>Bad: 你们叹息了一分钟，郑大妈也叹息了。 |
| multiple edits: 麻醉 -> 哭; 老虎 -> 一会儿; 麻醉 -> 哭 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他麻醉了老虎，她们也麻醉了。<br>Bad: 他哭了一会儿，她们也哭了。 |
| multiple edits: 麻醉 -> 微笑; 大象 -> 一天; 麻醉 -> 微笑 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你麻醉了大象，徐小姐也麻醉了。<br>Bad: 你微笑了一天，徐小姐也微笑了。 |
| multiple edits: 麻醉 -> 起飞; 大象 -> 一分钟; 麻醉 -> 起飞 | ellipsis_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们麻醉了大象，王小姐也麻醉了。<br>Bad: 我们起飞了一分钟，王小姐也起飞了。 |
| 串 → 个 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太买了五串香蕉，王大娘八串。<br>Bad: 何太太买了五串香蕉，王大娘八个。 |
| 串 → 块 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三吃了六串香蕉，冯大哥四串。<br>Bad: 张三吃了六串香蕉，冯大哥四块。 |
| 只 → 串 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨吃了七只鸡，赵大爷八只。<br>Bad: 王姨吃了七只鸡，赵大爷八串。 |
| 是 → 买给了冯大哥 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们买给了冯大哥六把椅子，我们的妈妈也是。<br>Bad: 他们买给了冯大哥六把椅子，我们的妈妈也买给了冯大哥。 |
| 是 → 买给了吴太太 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们买给了吴太太几条裙子，这个舞者也是。<br>Bad: 我们买给了吴太太几条裙子，这个舞者也买给了吴太太。 |
| 是 → 买给了宋女士 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷买给了宋女士好几百张桌子，张先生的爸爸也是。<br>Bad: 赵大爷买给了宋女士好几百张桌子，张先生的爸爸也买给了宋女士。 |
| 是 → 买给了张婶 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈买给了张婶几只手套，王大娘的下属也是。<br>Bad: 郑大妈买给了张婶几只手套，王大娘的下属也买给了张婶。 |
| 是 → 买给了杨大哥 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他买给了杨大哥十个收音机，那个顾客也是。<br>Bad: 他买给了杨大哥十个收音机，那个顾客也买给了杨大哥。 |
| 是 → 买给了王五 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥买给了王五两条裤子，张先生的下属也是。<br>Bad: 杨大哥买给了王五两条裤子，张先生的下属也买给了王五。 |
| 是 → 买给了王大娘 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他买给了王大娘几个玻璃珠，这个上级也是。<br>Bad: 他买给了王大娘几个玻璃珠，这个上级也买给了王大娘。 |
| 是 → 买给了王小姐 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他买给了王小姐一张桌子，王先生的学生也是。<br>Bad: 他买给了王小姐一张桌子，王先生的学生也买给了王小姐。 |
| 是 → 买给了郑大妈 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘买给了郑大妈三张桌子，那十个演奏员也是。<br>Bad: 王大娘买给了郑大妈三张桌子，那十个演奏员也买给了郑大妈。 |
| 是 → 买给了陈大姐 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们买给了陈大姐几只袜子，这个钢琴家也是。<br>Bad: 我们买给了陈大姐几只袜子，这个钢琴家也买给了陈大姐。 |
| 是 → 借给了小明 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三借给了小明许多本教材，这八位司机也是。<br>Bad: 张三借给了小明许多本教材，这八位司机也借给了小明。 |
| 是 → 借给了张婶 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘借给了张婶几本教材，另外十位顾客也是。<br>Bad: 王大娘借给了张婶几本教材，另外十位顾客也借给了张婶。 |
| 是 → 借给了杨大哥 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三借给了杨大哥几把椅子，陈大姐的爸爸也是。<br>Bad: 张三借给了杨大哥几把椅子，陈大姐的爸爸也借给了杨大哥。 |
| 是 → 借给了王大娘 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她借给了王大娘好几百把椅子，吴太太的女儿也是。<br>Bad: 她借给了王大娘好几百把椅子，吴太太的女儿也借给了王大娘。 |
| 是 → 借给了胡大爷 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我借给了胡大爷好几百个收音机，那位父亲也是。<br>Bad: 我借给了胡大爷好几百个收音机，那位父亲也借给了胡大爷。 |
| 是 → 卖给了宋女士 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们卖给了宋女士九个玻璃珠，李先生的兄弟也是。<br>Bad: 你们卖给了宋女士九个玻璃珠，李先生的兄弟也卖给了宋女士。 |
| 是 → 卖给了小明 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们卖给了小明四个充电器，你们的妈妈也是。<br>Bad: 你们卖给了小明四个充电器，你们的妈妈也卖给了小明。 |
| 是 → 卖给了张三 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们卖给了张三三只手套，这八位母亲也是。<br>Bad: 我们卖给了张三三只手套，这八位母亲也卖给了张三。 |
| 是 → 卖给了张先生 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈卖给了张先生两本教材，宋女士的同事也是。<br>Bad: 郑大妈卖给了张先生两本教材，宋女士的同事也卖给了张先生。 |
| 是 → 卖给了李先生 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生卖给了李先生两把椅子，那位舞者也是。<br>Bad: 张先生卖给了李先生两把椅子，那位舞者也卖给了李先生。 |
| 是 → 卖给了杨大哥 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太卖给了杨大哥七把椅子，他们的儿子也是。<br>Bad: 吴太太卖给了杨大哥七把椅子，他们的儿子也卖给了杨大哥。 |
| 是 → 卖给了王先生 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷卖给了王先生一张桌子，另外九个奴隶也是。<br>Bad: 胡大爷卖给了王先生一张桌子，另外九个奴隶也卖给了王先生。 |
| 是 → 卖给了王大娘 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她卖给了王大娘非常多本教材，陈大姐的下属也是。<br>Bad: 她卖给了王大娘非常多本教材，陈大姐的下属也卖给了王大娘。 |
| 是 → 卖给了陈大姐 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷卖给了陈大姐八个杯子，另外七个儿子也是。<br>Bad: 胡大爷卖给了陈大姐八个杯子，另外七个儿子也卖给了陈大姐。 |
| 是 → 寄给了吴太太 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐寄给了吴太太好几百本教材，另外五位顾客也是。<br>Bad: 王小姐寄给了吴太太好几百本教材，另外五位顾客也寄给了吴太太。 |
| 是 → 寄给了周大妈 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们寄给了周大妈好几十张桌子，那位舞者也是。<br>Bad: 我们寄给了周大妈好几十张桌子，那位舞者也寄给了周大妈。 |
| 是 → 寄给了宋女士 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太寄给了宋女士好几十只袜子，张三的老板也是。<br>Bad: 吴太太寄给了宋女士好几十只袜子，张三的老板也寄给了宋女士。 |
| 是 → 寄给了小王 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈寄给了小王好几本教材，这位父亲也是。<br>Bad: 郑大妈寄给了小王好几本教材，这位父亲也寄给了小王。 |
| 是 → 寄给了张夫人 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四寄给了张夫人许多把椅子，这两个音乐家也是。<br>Bad: 李四寄给了张夫人许多把椅子，这两个音乐家也寄给了张夫人。 |
| 是 → 寄给了张婶 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷寄给了张婶四只袜子，张先生的同事也是。<br>Bad: 胡大爷寄给了张婶四只袜子，张先生的同事也寄给了张婶。 |
| 是 → 寄给了徐小姐 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们寄给了徐小姐八张桌子，这两位空姐也是。<br>Bad: 你们寄给了徐小姐八张桌子，这两位空姐也寄给了徐小姐。 |
| 是 → 寄给了王姨 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们寄给了王姨五个饮料瓶，你们的下属也是。<br>Bad: 我们寄给了王姨五个饮料瓶，你们的下属也寄给了王姨。 |
| 是 → 寄给了赵大爷 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐寄给了赵大爷两条裙子，张三的妹妹也是。<br>Bad: 王小姐寄给了赵大爷两条裙子，张三的妹妹也寄给了赵大爷。 |
| 是 → 送给了周大妈 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你送给了周大妈许多把椅子，这七个记者也是。<br>Bad: 你送给了周大妈许多把椅子，这七个记者也送给了周大妈。 |
| 是 → 送给了张三 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他送给了张三许多张桌子，那十位工人也是。<br>Bad: 他送给了张三许多张桌子，那十位工人也送给了张三。 |
| 是 → 送给了徐小姐 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们送给了徐小姐四张桌子，这六个音乐家也是。<br>Bad: 他们送给了徐小姐四张桌子，这六个音乐家也送给了徐小姐。 |
| 是 → 送给了杨大哥 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生送给了杨大哥三只手套，那七个打工人也是。<br>Bad: 刘先生送给了杨大哥三只手套，那七个打工人也送给了杨大哥。 |
| 是 → 送给了赵大爷 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她送给了赵大爷十几把椅子，这七个钢琴家也是。<br>Bad: 她送给了赵大爷十几把椅子，这七个钢琴家也送给了赵大爷。 |
| 是 → 递给了冯大哥 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈递给了冯大哥三条裤子，另外四个工人也是。<br>Bad: 郑大妈递给了冯大哥三条裤子，另外四个工人也递给了冯大哥。 |
| 是 → 递给了小明 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们递给了小明四把椅子，另外九位父亲也是。<br>Bad: 她们递给了小明四把椅子，另外九位父亲也递给了小明。 |
| 是 → 递给了张三 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们递给了张三好几百本教材，那六位员工也是。<br>Bad: 你们递给了张三好几百本教材，那六位员工也递给了张三。 |
| 是 → 递给了张先生 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们递给了张先生四只袜子，那位音乐家也是。<br>Bad: 他们递给了张先生四只袜子，那位音乐家也递给了张先生。 |
| 是 → 递给了张夫人 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她递给了张夫人七只手套，这一个弟弟也是。<br>Bad: 她递给了张夫人七只手套，这一个弟弟也递给了张夫人。 |
| 是 → 递给了张婶 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四递给了张婶四本教材，王小姐的父亲也是。<br>Bad: 李四递给了张婶四本教材，王小姐的父亲也递给了张婶。 |
| 是 → 递给了李先生 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我递给了李先生三个杯子，我的上级也是。<br>Bad: 我递给了李先生三个杯子，我的上级也递给了李先生。 |
| 是 → 递给了王五 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们递给了王五三张桌子，那个弟弟也是。<br>Bad: 我们递给了王五三张桌子，那个弟弟也递给了王五。 |
| 是 → 递给了王大娘 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我递给了王大娘七个收音机，另外四个消费者也是。<br>Bad: 我递给了王大娘七个收音机，另外四个消费者也递给了王大娘。 |
| 是 → 递给了王姨 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐递给了王姨一个充电器，那个工人也是。<br>Bad: 王小姐递给了王姨一个充电器，那个工人也递给了王姨。 |
| 是 → 递给了郑大妈 | ellipsis_double_object | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们递给了郑大妈好几百本教材，另外五位下属也是。<br>Bad: 我们递给了郑大妈好几百本教材，另外五位下属也递给了郑大妈。 |
| 杯 → 串 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太买了五杯橙汁，张夫人一杯。<br>Bad: 吴太太买了五杯橙汁，张夫人一串。 |
| 桶 → 片 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生买了七桶啤酒，赵大爷九桶。<br>Bad: 张先生买了七桶啤酒，赵大爷九片。 |
| 桶 → 瓶 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨买了四桶方便面，陈大姐九桶。<br>Bad: 王姨买了四桶方便面，陈大姐九瓶。 |
| 片 → 块 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人吃了七片面包，徐小姐四片。<br>Bad: 张夫人吃了七片面包，徐小姐四块。 |
| 瓶 → 串 | ellipsis_n_bar_class | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐买了三瓶橙汁，王姨四瓶。<br>Bad: 徐小姐买了三瓶橙汁，王姨四串。 |

## fci_licensing

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| bad inserts 都 | fci_renhe_ruguo | 300 | 0.9967 | 0.0033 | +0.9933 | 0.0000 | Good: 如果有任何人憎恨打工人，马上告诉我。<br>Bad: 如果有任何人都憎恨打工人，马上告诉我。 |
| 任何 → 有些 | fci_renhe_subj | 300 | 0.7867 | 0.0200 | +0.7667 | 0.0000 | Good: 任何人都会开火车。<br>Bad: 有些人都会开火车。 |
| 所有 → 任何 | fci_renhe_suoyou | 300 | 0.8333 | 0.6867 | +0.1467 | 0.0000 | Good: 所有人都驾驶过火车了。<br>Bad: 任何人都驾驶过火车了。 |
| bad deletes 都 | fci_renhe_dou | 300 | 0.7667 | 0.8867 | -0.1200 | 0.0000 | Good: 任何人都可以驾驶货车。<br>Bad: 任何人可以驾驶货车。 |
| 任何 → 没有 | fci_renhe_prepP | 300 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他可以跟任何人一起呼吸。<br>Bad: 他可以跟没有人一起呼吸。 |

## nominal_expression

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad inserts 普通; bad deletes 普通 | you_quantifier_adj | 3 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们很冷静，有了那位普通的空姐。<br>Bad: 他们很冷静，有了普通那位的空姐。 |
| 他 → 工人 | PN_numP_a | 3 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们这六个人检查了肚子。<br>Bad: 工人们这六个人检查了肚子。 |
| 她 → 爸爸 | PN_numP_a | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们那五个人把牛屠宰了。<br>Bad: 爸爸们那五个人把牛屠宰了。 |
| 她 → 舞者 | PN_numP_a | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们那四个人吃过蛋糕了。<br>Bad: 舞者们那四个人吃过蛋糕了。 |
| 我 → 妹妹 | PN_numP_a | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那九个人盖过被子了。<br>Bad: 妹妹们那九个人盖过被子了。 |
| 我 → 姐姐 | PN_numP_a | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那十个人把鱼炖了。<br>Bad: 姐姐们那十个人把鱼炖了。 |
| 我 → 演员 | PN_numP_a | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那两个人把鱼清蒸了。<br>Bad: 演员们那两个人把鱼清蒸了。 |
| multiple edits: bad inserts 特殊; bad deletes 特殊 | you_quantifier_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张三非常高兴，有了这位特殊的工人。<br>Bad: 张三非常高兴，有了特殊这位的工人。 |
| 他 → 哥哥 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们那两个人把鼻子打断了。<br>Bad: 哥哥们那两个人把鼻子打断了。 |
| 他 → 消防员 | PN_numP_a | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们这两个人唱了小调。<br>Bad: 消防员们这两个人唱了小调。 |
| 他 → 空姐 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们这十个人盖过被子了。<br>Bad: 空姐们这十个人盖过被子了。 |
| 你 → 妈妈 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们那八个人麻醉过老虎了。<br>Bad: 妈妈们那八个人麻醉过老虎了。 |
| 你 → 妹妹 | PN_numP_a | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们那两个人包扎了腿。<br>Bad: 妹妹们那两个人包扎了腿。 |
| 你 → 爸爸 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们那九个人吹过笛子了。<br>Bad: 爸爸们那九个人吹过笛子了。 |
| 你 → 老师 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们这七个人把飞机驾驶了。<br>Bad: 老师们这七个人把飞机驾驶了。 |
| 你 → 老板 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们那七个人创作过小说了。<br>Bad: 老板们那七个人创作过小说了。 |
| 她 → 儿子 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们那六个人把飞机开了。<br>Bad: 儿子们那六个人把飞机开了。 |
| 她 → 罪犯 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们那五个人把被子盖了。<br>Bad: 罪犯们那五个人把被子盖了。 |
| 她 → 老师 | PN_numP_a | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们这三个人跨越过海洋了。<br>Bad: 老师们这三个人跨越过海洋了。 |
| 她 → 飞行员 | PN_numP_a | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们那五个人检查了胃。<br>Bad: 飞行员们那五个人检查了胃。 |
| 我 → 女儿 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那三个人把鸭烧了。<br>Bad: 女儿们那三个人把鸭烧了。 |
| 我 → 奴隶 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那十个人看过录像带了。<br>Bad: 奴隶们那十个人看过录像带了。 |
| 我 → 学生 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那七个人把大提琴拉了。<br>Bad: 学生们那七个人把大提琴拉了。 |
| 我 → 服务员 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那六个人炖过鸭了。<br>Bad: 服务员们那六个人炖过鸭了。 |
| 我 → 母亲 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那五个人炖了鸭。<br>Bad: 母亲们那五个人炖了鸭。 |
| 我 → 演奏员 | PN_numP_a | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们那六个人把方便面吃了。<br>Bad: 演奏员们那六个人把方便面吃了。 |
| 她 → 工人 | PN_numP_a | 6 | 1.0000 | 0.1667 | +0.8333 | 0.0000 | Good: 她们这九个人捕捉了大象。<br>Bad: 工人们这九个人捕捉了大象。 |
| 他 → 钢琴家 | PN_numP_a | 4 | 0.7500 | 0.0000 | +0.7500 | 0.0000 | Good: 他们那三个人打断过鼻子了。<br>Bad: 钢琴家们那三个人打断过鼻子了。 |
| 你 → 记者 | PN_numP_a | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 你们这三个人喝了红酒。<br>Bad: 记者们这三个人喝了红酒。 |
| 我 → 姐妹 | PN_numP_a | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 我们这十个人跨越过沙漠了。<br>Bad: 姐妹们这十个人跨越过沙漠了。 |
| 他 → 演员 | PN_numP_a | 6 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 他们这三个人把鱼炖了。<br>Bad: 演员们这三个人把鱼炖了。 |
| 他 → 音乐家 | PN_numP_a | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 他们这两个人清洗过杯子了。<br>Bad: 音乐家们这两个人清洗过杯子了。 |
| 她 → 妹妹 | PN_numP_a | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 她们那两个人领养了小狗。<br>Bad: 妹妹们那两个人领养了小狗。 |
| 她 → 老板 | PN_numP_a | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 她们那五个人麻醉了老虎。<br>Bad: 老板们那五个人麻醉了老虎。 |
| 我 → 哥哥 | PN_numP_a | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 我们那八个人把双簧吹了。<br>Bad: 哥哥们那八个人把双簧吹了。 |
| 我 → 老板 | PN_numP_a | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 我们那六个人把啤酒喝了。<br>Bad: 老板们那六个人把啤酒喝了。 |
| 我 → 音乐家 | PN_numP_a | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 我们这六个人驾驶了火车。<br>Bad: 音乐家们这六个人驾驶了火车。 |
| 你 → 演员 | PN_numP_a | 7 | 0.1429 | 0.7143 | -0.5714 | 0.0000 | Good: 你们这六个人炖了鸭。<br>Bad: 演员们这六个人炖了鸭。 |
| bad inserts 有 | you_yige | 300 | 0.8967 | 0.3933 | +0.5033 | 0.0000 | Good: 那条鱼是几位学生烧的。<br>Bad: 那条鱼是有几位学生烧的。 |
| 他 → 姐妹 | PN_numP_a | 4 | 0.2500 | 0.7500 | -0.5000 | 0.0000 | Good: 他们这八个人把坚果吃了。<br>Bad: 姐妹们这八个人把坚果吃了。 |
| 他 → 姐姐 | PN_numP_a | 4 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 他们那四个人把鸡爆炒了。<br>Bad: 姐姐们那四个人把鸡爆炒了。 |
| 他 → 消费者 | PN_numP_a | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他们这五个人把脚打断了。<br>Bad: 消费者们这五个人把脚打断了。 |
| 他 → 演奏员 | PN_numP_a | 4 | 0.2500 | 0.7500 | -0.5000 | 0.0000 | Good: 他们那两个人把货车开了。<br>Bad: 演奏员们那两个人把货车开了。 |
| 你 → 服务员 | PN_numP_a | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你们那十个人清蒸了鸡。<br>Bad: 服务员们那十个人清蒸了鸡。 |
| multiple edits: bad inserts 年老; bad deletes 年老 | you_quantifier_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 小王有点苦恼，有了这个年老的妹妹。<br>Bad: 小王有点苦恼，有了年老这个的妹妹。 |
| multiple edits: bad inserts 浅显; bad deletes 浅显 | you_quantifier_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 张婶有点困惑，有了这个浅显的态度。<br>Bad: 张婶有点困惑，有了浅显这个的态度。 |
| multiple edits: bad inserts 深刻; bad deletes 深刻 | you_quantifier_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他有点高兴，有了那个深刻的理念。<br>Bad: 他有点高兴，有了深刻那个的理念。 |
| multiple edits: bad inserts 胖; bad deletes 胖 | you_quantifier_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你很开心，有了这位胖的工人。<br>Bad: 你很开心，有了胖这位的工人。 |
| 他 → 奴隶 | PN_numP_a | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他们这六个人把教材预习了。<br>Bad: 奴隶们这六个人把教材预习了。 |
| 他 → 顾客 | PN_numP_a | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 他们那九个人把老虎捕捉了。<br>Bad: 顾客们那九个人把老虎捕捉了。 |
| 他 → 飞行员 | PN_numP_a | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他们这三个人爆炒了鱼。<br>Bad: 飞行员们这三个人爆炒了鱼。 |
| 你 → 母亲 | PN_numP_a | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 你们那八个人创作过漫画了。<br>Bad: 母亲们那八个人创作过漫画了。 |
| 你 → 舞者 | PN_numP_a | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 你们这两个人把手打断了。<br>Bad: 舞者们这两个人把手打断了。 |
| 你 → 顾客 | PN_numP_a | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 你们那五个人烧了鸡。<br>Bad: 顾客们那五个人烧了鸡。 |
| 你 → 领导 | PN_numP_a | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你们那三个人制作了电视剧。<br>Bad: 领导们那三个人制作了电视剧。 |
| 她 → 消费者 | PN_numP_a | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 她们这十个人把电影拍摄了。<br>Bad: 消费者们这十个人把电影拍摄了。 |
| 她 → 演员 | PN_numP_a | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 她们那七个人把玻璃珠弹了。<br>Bad: 演员们那七个人把玻璃珠弹了。 |
| 她 → 钢琴家 | PN_numP_a | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 她们这七个人弹过古筝了。<br>Bad: 钢琴家们这七个人弹过古筝了。 |
| 我 → 司机 | PN_numP_a | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 我们那九个人检查了手。<br>Bad: 司机们那九个人检查了手。 |
| 我 → 同事 | PN_numP_a | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 我们那九个人把手打断了。<br>Bad: 同事们那九个人把手打断了。 |
| 我 → 罪犯 | PN_numP_a | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 我们这九个人预习过教材了。<br>Bad: 罪犯们这九个人预习过教材了。 |
| multiple edits: bad inserts 八个; bad deletes 八个 | PN_numP_b | 38 | 0.6053 | 0.1842 | +0.4211 | 0.0000 | Good: 王大娘对她们八个非常好。<br>Bad: 王大娘对八个她们非常好。 |
| multiple edits: bad inserts 热; bad deletes 热 | you_quantifier_adj | 24 | 0.4583 | 0.8750 | -0.4167 | 0.0000 | Good: 他们比较愤怒，有了那桶热的矿泉水。<br>Bad: 他们比较愤怒，有了热那桶的矿泉水。 |
| 你 → 工人 | PN_numP_a | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 你们那八个人看过录像带了。<br>Bad: 工人们那八个人看过录像带了。 |
| multiple edits: bad inserts 十个; bad deletes 十个 | PN_numP_b | 38 | 0.6316 | 1.0000 | -0.3684 | 0.0000 | Good: 刘先生对我们十个比较好。<br>Bad: 刘先生对十个我们比较好。 |
| 她 → 上级 | PN_numP_a | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 她们这三个人观看了视频。<br>Bad: 上级们这三个人观看了视频。 |
| 她 → 吉他手 | PN_numP_a | 6 | 0.5000 | 0.1667 | +0.3333 | 0.0000 | Good: 她们那三个人检查过肚子了。<br>Bad: 吉他手们那三个人检查过肚子了。 |
| multiple edits: bad inserts 瘦; bad deletes 瘦 | you_quantifier_adj | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 她们有点困惑，有了那位瘦的学生。<br>Bad: 她们有点困惑，有了瘦那位的学生。 |
| multiple edits: bad inserts 高; bad deletes 高 | you_quantifier_adj | 6 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 徐小姐有点高兴，有了那位高的下属。<br>Bad: 徐小姐有点高兴，有了高那位的下属。 |
| 你 → 员工 | PN_numP_a | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 你们这五个人把手账制作了。<br>Bad: 员工们这五个人把手账制作了。 |
| 我 → 飞行员 | PN_numP_a | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 我们那四个人把卡车开了。<br>Bad: 飞行员们那四个人把卡车开了。 |
| multiple edits: bad inserts 六个; bad deletes 六个 | PN_numP_b | 24 | 0.5417 | 0.2083 | +0.3333 | 0.0000 | Good: 徐小姐对你们六个非常好。<br>Bad: 徐小姐对六个你们非常好。 |
| multiple edits: bad inserts 应该; bad deletes 应该 | nominal_modal_insertion | 70 | 0.9571 | 0.6286 | +0.3286 | 0.0000 | Good: 她们四位母亲应该品茶。<br>Bad: 她们应该四位母亲品茶。 |
| multiple edits: bad inserts 七个; bad deletes 七个 | PN_numP_b | 34 | 0.5882 | 0.2941 | +0.2941 | 0.0000 | Good: 胡大爷对你们七个非常好。<br>Bad: 胡大爷对七个你们非常好。 |
| multiple edits: bad inserts 两个; bad deletes 两个 | PN_numP_b | 34 | 0.5294 | 0.8235 | -0.2941 | 0.0000 | Good: 吴太太对我们两个很不好。<br>Bad: 吴太太对两个我们很不好。 |
| 他 → 打工人 | PN_numP_a | 7 | 0.8571 | 0.5714 | +0.2857 | 0.0000 | Good: 他们这八个人清蒸过鸡了。<br>Bad: 打工人们这八个人清蒸过鸡了。 |
| multiple edits: bad inserts 九个; bad deletes 九个 | PN_numP_b | 33 | 0.4848 | 0.2121 | +0.2727 | 0.0000 | Good: 王大娘对我们九个很好。<br>Bad: 王大娘对九个我们很好。 |
| multiple edits: bad inserts 四个; bad deletes 四个 | PN_numP_b | 35 | 0.7714 | 0.5143 | +0.2571 | 0.0000 | Good: 冯大哥对我们四个有点好。<br>Bad: 冯大哥对四个我们有点好。 |
| 他 → 下属 | PN_numP_a | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 他们这八个人爆炒过鱼了。<br>Bad: 下属们这八个人爆炒过鱼了。 |
| 他 → 母亲 | PN_numP_a | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 他们那六个人创作了小说。<br>Bad: 母亲们那六个人创作了小说。 |
| 她 → 司机 | PN_numP_a | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 她们那两个人开过飞机了。<br>Bad: 司机们那两个人开过飞机了。 |
| 她 → 同事 | PN_numP_a | 4 | 0.2500 | 0.0000 | +0.2500 | 0.0000 | Good: 她们那十个人领养了小猫。<br>Bad: 同事们那十个人领养了小猫。 |
| 她 → 打工人 | PN_numP_a | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 她们那十个人制作了电影。<br>Bad: 打工人们那十个人制作了电影。 |
| 她 → 记者 | PN_numP_a | 4 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 她们这四个人把飞机开了。<br>Bad: 记者们这四个人把飞机开了。 |
| multiple edits: bad inserts 五个; bad deletes 五个 | PN_numP_b | 34 | 0.7353 | 0.5000 | +0.2353 | 0.0000 | Good: 杨大哥对她们五个很不好。<br>Bad: 杨大哥对五个她们很不好。 |
| multiple edits: bad inserts 三个; bad deletes 三个 | PN_numP_b | 30 | 0.8000 | 0.6000 | +0.2000 | 0.0000 | Good: 张三对我们三个非常不好。<br>Bad: 张三对三个我们非常不好。 |
| multiple edits: bad inserts 吉; bad inserts 手 | PN_numP_a | 5 | 0.4000 | 0.2000 | +0.2000 | 0.0000 | Good: 他们这三个人炖了鱼。<br>Bad: 吉他手们这三个人炖了鱼。 |
| multiple edits: bad inserts 小; bad deletes 小 | you_quantifier_adj | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 她非常失望，有了那个小的头。<br>Bad: 她非常失望，有了小那个的头。 |
| multiple edits: bad inserts 咸; bad deletes 咸 | you_quantifier_adj | 23 | 1.0000 | 0.8261 | +0.1739 | 0.0000 | Good: 周大妈很快乐，有了那串咸的香蕉。<br>Bad: 周大妈很快乐，有了咸那串的香蕉。 |
| multiple edits: bad inserts 淡; bad deletes 淡 | you_quantifier_adj | 24 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 你很伤心，有了那串淡的香蕉。<br>Bad: 你很伤心，有了淡那串的香蕉。 |
| multiple edits: bad inserts 不可以; bad deletes 不可以 | nominal_modal_insertion | 76 | 0.5789 | 0.6711 | -0.0921 | 0.0000 | Good: 他一位父亲不可以看戏。<br>Bad: 他不可以一位父亲看戏。 |
| multiple edits: bad inserts 大; bad deletes 大 | you_quantifier_adj | 13 | 0.6154 | 0.6923 | -0.0769 | 0.0000 | Good: 王小姐比较快乐，有了一只大的脚。<br>Bad: 王小姐比较快乐，有了大一只的脚。 |
| multiple edits: bad inserts 温驯; bad deletes 温驯 | you_quantifier_adj | 15 | 0.9333 | 1.0000 | -0.0667 | 0.0000 | Good: 王小姐有点愤怒，有了这条温驯的鱼。<br>Bad: 王小姐有点愤怒，有了温驯这条的鱼。 |
| bad deletes 们 | singular_PN_but_plural_pron | 300 | 0.0333 | 0.0967 | -0.0633 | 0.0000 | Good: 王小姐约束了宋女士她们两个。<br>Bad: 王小姐约束了宋女士她两个。 |
| 和 → 兼 | noun_phrase_conjunction_jian | 158 | 0.5506 | 0.6013 | -0.0506 | 0.0000 | Good: 那位服务员嫌弃那位父亲和我。<br>Bad: 那位服务员嫌弃那位父亲兼我。 |
| bad inserts 们 | plural_cardinal_men_b | 300 | 0.0400 | 0.0867 | -0.0467 | 0.0000 | Good: 我见到了九个奴隶。<br>Bad: 我见到了九个奴隶们。 |
| multiple edits: bad inserts 冷; bad deletes 冷 | you_quantifier_adj | 22 | 1.0000 | 0.9545 | +0.0455 | 0.0000 | Good: 你们比较快乐，有了一瓶冷的红酒。<br>Bad: 你们比较快乐，有了冷一瓶的红酒。 |
| multiple edits: bad inserts 昂贵; bad deletes 昂贵 | you_quantifier_adj | 69 | 1.0000 | 0.9565 | +0.0435 | 0.0000 | Good: 王五有点苦恼，有了一个昂贵的充电器。<br>Bad: 王五有点苦恼，有了昂贵一个的充电器。 |
| bad inserts 们 | nominal_definite_men | 300 | 0.1100 | 0.0933 | +0.0167 | 0.0000 | Good: 没有记者检查手。<br>Bad: 没有记者们检查手。 |
| multiple edits: bad inserts 不应该; bad deletes 不应该 | nominal_modal_insertion | 67 | 0.5970 | 0.5821 | +0.0149 | 0.0000 | Good: 他们六个弟弟不应该来。<br>Bad: 他们不应该六个弟弟来。 |
| 跟 → 兼 | noun_phrase_conjunction_jian | 142 | 0.6761 | 0.6901 | -0.0141 | 0.0000 | Good: 小王的姐妹憎恨你跟他。<br>Bad: 小王的姐妹憎恨你兼他。 |
| multiple edits: bad inserts 可以; bad deletes 可以 | nominal_modal_insertion | 87 | 0.5977 | 0.6092 | -0.0115 | 0.0000 | Good: 他们两位父亲可以闲逛。<br>Bad: 他们可以两位父亲闲逛。 |
| bad deletes 是 | noun_adjective_shi | 300 | 0.8433 | 0.8367 | +0.0067 | 0.0000 | Good: 我们是空姐。<br>Bad: 我们空姐。 |
| bad inserts 们 | plural_cardinal_men_a | 300 | 0.0867 | 0.0900 | -0.0033 | 0.0000 | Good: 昨天旅行来了一位司机。<br>Bad: 昨天旅行来了一位司机们。 |
| multiple edits: bad inserts 便宜; bad deletes 便宜 | you_quantifier_adj | 55 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我比较高兴，有了一瓶便宜的矿泉水。<br>Bad: 我比较高兴，有了便宜一瓶的矿泉水。 |
| multiple edits: bad inserts 凶猛; bad deletes 凶猛 | you_quantifier_adj | 17 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我有点高兴，有了一头凶猛的大象。<br>Bad: 我有点高兴，有了凶猛一头的大象。 |
| multiple edits: bad inserts 年轻; bad deletes 年轻 | you_quantifier_adj | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生比较开心，有了那位年轻的老师。<br>Bad: 王先生比较开心，有了年轻那位的老师。 |
| 他 → 司机 | PN_numP_a | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们那九个人炖过鸭了。<br>Bad: 司机们那九个人炖过鸭了。 |
| 她 → 顾客 | PN_numP_a | 5 | 0.4000 | 0.4000 | +0.0000 | 0.0000 | Good: 她们那三个人预习过教材了。<br>Bad: 顾客们那三个人预习过教材了。 |
| 我 → 吉他手 | PN_numP_a | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们那六个人捕捉过老虎了。<br>Bad: 吉他手们那六个人捕捉过老虎了。 |
| 我 → 消费者 | PN_numP_a | 5 | 0.6000 | 0.6000 | +0.0000 | 0.0000 | Good: 我们那八个人预习了教材。<br>Bad: 消费者们那八个人预习了教材。 |
| 你 → 钢琴家 | PN_numP_a | 4 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 你们这三个人检查了头。<br>Bad: 钢琴家们这三个人检查了头。 |
| multiple edits: bad inserts 矮; bad deletes 矮 | you_quantifier_adj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐有点失望，有了一位矮的服务员。<br>Bad: 徐小姐有点失望，有了矮一位的服务员。 |
| 他 → 上级 | PN_numP_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们这五个人看了手账。<br>Bad: 上级们这五个人看了手账。 |
| 他 → 弟弟 | PN_numP_a | 3 | 0.3333 | 0.3333 | +0.0000 | 0.0000 | Good: 他们那八个人盖了被子。<br>Bad: 弟弟们那八个人盖了被子。 |
| 他 → 警察 | PN_numP_a | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 他们那四个人演奏过奏鸣曲了。<br>Bad: 警察们那四个人演奏过奏鸣曲了。 |
| 你 → 打工人 | PN_numP_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这四个人把小提琴拉了。<br>Bad: 打工人们这四个人把小提琴拉了。 |
| 你 → 音乐家 | PN_numP_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们那三个人麻醉了老虎。<br>Bad: 音乐家们那三个人麻醉了老虎。 |
| 她 → 兄弟 | PN_numP_a | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 她们那十个人观看过电影了。<br>Bad: 兄弟们那十个人观看过电影了。 |
| 她 → 演奏员 | PN_numP_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们那十个人包扎了肚子。<br>Bad: 演奏员们那十个人包扎了肚子。 |
| 我 → 记者 | PN_numP_a | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 我们那三个人把鸭清蒸了。<br>Bad: 记者们那三个人把鸭清蒸了。 |
| 我 → 钢琴家 | PN_numP_a | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 我们那七个人把鼻子检查了。<br>Bad: 钢琴家们那七个人把鼻子检查了。 |
| multiple edits: bad inserts 丑; bad deletes 丑 | you_quantifier_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王非常伤心，有了那位丑的服务员。<br>Bad: 小王非常伤心，有了丑那位的服务员。 |
| 他 → 服务员 | PN_numP_a | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们那十个人把火车驾驶了。<br>Bad: 服务员们那十个人把火车驾驶了。 |
| 他 → 老师 | PN_numP_a | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们那七个人检查过腿了。<br>Bad: 老师们那七个人检查过腿了。 |
| 他 → 老板 | PN_numP_a | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们这九个人把沙漠跨越了。<br>Bad: 老板们这九个人把沙漠跨越了。 |
| 他 → 记者 | PN_numP_a | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 他们那三个人捕捉过鱼了。<br>Bad: 记者们那三个人捕捉过鱼了。 |
| 他 → 领导 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们这八个人唱过歌了。<br>Bad: 领导们这八个人唱过歌了。 |
| 你 → 司机 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们那十个人煮了鸭。<br>Bad: 司机们那十个人煮了鸭。 |
| 你 → 奴隶 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这十个人炖了鸡。<br>Bad: 奴隶们这十个人炖了鸡。 |
| 你 → 姐妹 | PN_numP_a | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 你们这三个人把玻璃珠弹了。<br>Bad: 姐妹们这三个人把玻璃珠弹了。 |
| 她 → 下属 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们那八个人捕捉了老虎。<br>Bad: 下属们那八个人捕捉了老虎。 |
| 她 → 姐妹 | PN_numP_a | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 她们这七个人领养了小狗。<br>Bad: 姐妹们这七个人领养了小狗。 |
| 她 → 服务员 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们那两个人拍摄过电影了。<br>Bad: 服务员们那两个人拍摄过电影了。 |
| 她 → 音乐家 | PN_numP_a | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 她们这五个人唱过美声了。<br>Bad: 音乐家们这五个人唱过美声了。 |
| 她 → 领导 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们这两个人把牛屠宰了。<br>Bad: 领导们这两个人把牛屠宰了。 |
| 我 → 小孩 | PN_numP_a | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们这五个人把鸭爆炒了。<br>Bad: 小孩们这五个人把鸭爆炒了。 |
| 我 → 工人 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们这八个人炖了鸭。<br>Bad: 工人们这八个人炖了鸭。 |
| 我 → 消防员 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们那三个人看过录像带了。<br>Bad: 消防员们那三个人看过录像带了。 |
| 我 → 警察 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们这八个人创作过小说了。<br>Bad: 警察们这八个人创作过小说了。 |
| 我 → 领导 | PN_numP_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们那三个人清洗了杯子。<br>Bad: 领导们那三个人清洗了杯子。 |
| multiple edits: bad inserts 保守; bad deletes 保守 | you_quantifier_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他非常伤心，有了那个保守的态度。<br>Bad: 他非常伤心，有了保守那个的态度。 |
| 他 → 兄弟 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们那三个人屠宰了牛。<br>Bad: 兄弟们那三个人屠宰了牛。 |
| 他 → 女儿 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们这十个人煮了鸭。<br>Bad: 女儿们这十个人煮了鸭。 |
| 他 → 妹妹 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们这两个人唱过戏曲了。<br>Bad: 妹妹们这两个人唱过戏曲了。 |
| 他 → 学生 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们那七个人把被子盖了。<br>Bad: 学生们那七个人把被子盖了。 |
| 他 → 小孩 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们那九个人把红茶喝了。<br>Bad: 小孩们那九个人把红茶喝了。 |
| 他 → 朋友 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们这十个人打断过手了。<br>Bad: 朋友们这十个人打断过手了。 |
| 你 → 儿子 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这五个人弹了玻璃珠。<br>Bad: 儿子们这五个人弹了玻璃珠。 |
| 你 → 兄弟 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们这四个人领养了小猫。<br>Bad: 兄弟们这四个人领养了小猫。 |
| 你 → 吉他手 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这七个人烧过鸭了。<br>Bad: 吉他手们这七个人烧过鸭了。 |
| 你 → 女儿 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们那九个人演奏了奏鸣曲。<br>Bad: 女儿们那九个人演奏了奏鸣曲。 |
| 你 → 学生 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们那四个人清洗了杯子。<br>Bad: 学生们那四个人清洗了杯子。 |
| 你 → 朋友 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这三个人把鸡炖了。<br>Bad: 朋友们这三个人把鸡炖了。 |
| 你 → 演奏员 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这三个人喝过啤酒了。<br>Bad: 演奏员们这三个人喝过啤酒了。 |
| 你 → 罪犯 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们那七个人开了火车。<br>Bad: 罪犯们那七个人开了火车。 |
| 你 → 飞行员 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们这三个人把沙漠跨越了。<br>Bad: 飞行员们这三个人把沙漠跨越了。 |
| 她 → 员工 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们那十个人演奏了狂想曲。<br>Bad: 员工们那十个人演奏了狂想曲。 |
| 她 → 姐姐 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们这四个人创作过小说了。<br>Bad: 姐姐们这四个人创作过小说了。 |
| 她 → 小孩 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们这四个人盖过被子了。<br>Bad: 小孩们这四个人盖过被子了。 |
| 她 → 朋友 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们那三个人盖过被子了。<br>Bad: 朋友们那三个人盖过被子了。 |
| 她 → 父亲 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们那五个人开了卡车。<br>Bad: 父亲们那五个人开了卡车。 |
| 她 → 警察 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们这七个人喝过红酒了。<br>Bad: 警察们这七个人喝过红酒了。 |
| 我 → 下属 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们这八个人把被子盖了。<br>Bad: 下属们这八个人把被子盖了。 |
| 我 → 员工 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们这七个人把鸭爆炒了。<br>Bad: 员工们这七个人把鸭爆炒了。 |
| 我 → 朋友 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们这八个人喝过矿泉水了。<br>Bad: 朋友们这八个人喝过矿泉水了。 |
| 我 → 空姐 | PN_numP_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们这六个人把狂想曲演奏了。<br>Bad: 空姐们这六个人把狂想曲演奏了。 |
| 我 → 舞者 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们那十个人制作了动作片。<br>Bad: 舞者们那十个人制作了动作片。 |
| 我 → 顾客 | PN_numP_a | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们这六个人吹过双簧了。<br>Bad: 顾客们这六个人吹过双簧了。 |

## npi_licensing

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| 一个 → 任何 | npi_renhe_wh_question_subj | 9 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 一个演奏员会辩护哪些东西？<br>Bad: 任何演奏员会辩护哪些东西？ |
| 几位 → 任何 | npi_renhe_wh_question_subj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 几位消费者会支持什么？<br>Bad: 任何消费者会支持什么？ |
| 六位 → 任何 | npi_renhe_wh_question_subj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 六位空姐会知道什么？<br>Bad: 任何空姐会知道什么？ |
| multiple edits: bad inserts 你有; bad deletes 你的 | renhe_no_superordinate_negation | 300 | 0.9833 | 0.0467 | +0.9367 | 0.0000 | Good: 张三没有递给过小明任何你的葡萄。<br>Bad: 张三没有递给过小明你有任何葡萄。 |
| 十位 → 任何 | npi_renhe_wh_question_subj | 9 | 1.0000 | 0.1111 | +0.8889 | 0.0000 | Good: 十位消费者会批评哪些东西？<br>Bad: 任何消费者会批评哪些东西？ |
| 七位 → 任何 | npi_renhe_wh_question_subj | 4 | 0.7500 | 0.0000 | +0.7500 | 0.0000 | Good: 七位吉他手会知道什么东西？<br>Bad: 任何吉他手会知道什么东西？ |
| 一位 → 任何 | npi_renhe_wh_question_subj | 6 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 一位母亲会埋怨什么？<br>Bad: 任何母亲会埋怨什么？ |
| multiple edits: bad deletes 没有; bad inserts 没有 | npi_renhe_neg_scope_locP | 300 | 0.8933 | 0.2567 | +0.6367 | 0.0000 | Good: 这位领导没有在任何地点清洗过杯子。<br>Bad: 这位领导在任何地点没有清洗过杯子。 |
| 九位 → 任何 | npi_renhe_wh_question_subj | 5 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 九位老师会赞成什么？<br>Bad: 任何老师会赞成什么？ |
| 几个 → 任何 | npi_renhe_wh_question_subj | 7 | 0.8571 | 0.2857 | +0.5714 | 0.0000 | Good: 几个消费者会烧哪些东西？<br>Bad: 任何消费者会烧哪些东西？ |
| 这位 → 任何 | npi_renhe_wh_question_subj | 43 | 0.7209 | 0.2093 | +0.5116 | 0.0000 | Good: 这位顾客会鼓励什么？<br>Bad: 任何顾客会鼓励什么？ |
| 五位 → 任何 | npi_renhe_wh_question_subj | 6 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 五位消费者会尊重哪些东西？<br>Bad: 任何消费者会尊重哪些东西？ |
| multiple edits: bad deletes 任何; 她 -> 任何人 | npi_renhe_conditional | 99 | 0.9899 | 0.5354 | +0.4545 | 0.0000 | Good: 如果有任何人驾驶轮船，你就告诉她。<br>Bad: 如果有人驾驶轮船，你就告诉任何人。 |
| 那位 → 任何 | npi_renhe_wh_question_subj | 57 | 0.7544 | 0.3158 | +0.4386 | 0.0000 | Good: 那位顾客会赞成什么东西？<br>Bad: 任何顾客会赞成什么东西？ |
| 觉得 → 知道 | renhe_non_factive_verb | 55 | 1.0000 | 0.5636 | +0.4364 | 0.0000 | Good: 徐小姐不觉得打碎杯子对打工人有任何影响。<br>Bad: 徐小姐不知道打碎杯子对打工人有任何影响。 |
| 两位 → 任何 | npi_renhe_wh_question_subj | 7 | 0.8571 | 0.4286 | +0.4286 | 0.0000 | Good: 两位服务员会领养什么？<br>Bad: 任何服务员会领养什么？ |
| 十个 → 任何 | npi_renhe_wh_question_subj | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 十个舞者会憎恨什么东西？<br>Bad: 任何舞者会憎恨什么东西？ |
| 四位 → 任何 | npi_renhe_wh_question_subj | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 四位音乐家会伤害哪些东西？<br>Bad: 任何音乐家会伤害哪些东西？ |
| 九个 → 任何 | npi_renhe_wh_question_subj | 4 | 0.2500 | 0.0000 | +0.2500 | 0.0000 | Good: 九个演奏员会重建哪些东西？<br>Bad: 任何演奏员会重建哪些东西？ |
| 她的 → 任何 | npi_renhe_wh_question_obj | 89 | 0.4719 | 0.7191 | -0.2472 | 0.0000 | Good: 谁期待清蒸她的鱼？<br>Bad: 谁期待清蒸任何鱼？ |
| 希望 → 知道 | renhe_non_factive_verb | 41 | 1.0000 | 0.7561 | +0.2439 | 0.0000 | Good: 宋女士不希望开轮船对钢琴家有任何影响。<br>Bad: 宋女士不知道开轮船对钢琴家有任何影响。 |
| multiple edits: bad deletes 有没有; bad inserts 了; ？ -> 。 | npi_renhe_A_not_A_question | 300 | 0.8667 | 0.6433 | +0.2233 | 0.0000 | Good: 这个舞者在沙漠有没有演奏任何华尔兹？<br>Bad: 这个舞者在沙漠演奏了任何华尔兹。 |
| multiple edits: bad deletes 没; bad inserts 了 | renhe_no_episodic_sentences | 300 | 0.9767 | 0.7700 | +0.2067 | 0.0000 | Good: 过去张婶没看见任何人在制作电视剧。<br>Bad: 过去张婶看见了任何人在制作电视剧。 |
| 两个 → 任何 | npi_renhe_wh_question_subj | 10 | 0.4000 | 0.2000 | +0.2000 | 0.0000 | Good: 两个哥哥会包扎什么东西？<br>Bad: 任何哥哥会包扎什么东西？ |
| 七个 → 任何 | npi_renhe_wh_question_subj | 5 | 0.0000 | 0.2000 | -0.2000 | 0.0000 | Good: 七个司机会拍摄哪些东西？<br>Bad: 任何司机会拍摄哪些东西？ |
| 三位 → 任何 | npi_renhe_wh_question_subj | 5 | 0.4000 | 0.6000 | -0.2000 | 0.0000 | Good: 三位空姐会批评什么东西？<br>Bad: 任何空姐会批评什么东西？ |
| 你的 → 任何 | npi_renhe_wh_question_obj | 67 | 0.4627 | 0.6119 | -0.1493 | 0.0000 | Good: 谁不期待领养你的小猫？<br>Bad: 谁不期待领养任何小猫？ |
| 八个 → 任何 | npi_renhe_wh_question_subj | 8 | 0.1250 | 0.2500 | -0.1250 | 0.0000 | Good: 八个司机会反感哪些东西？<br>Bad: 任何司机会反感哪些东西？ |
| 这个 → 任何 | npi_renhe_wh_question_subj | 48 | 0.7500 | 0.6458 | +0.1042 | 0.0000 | Good: 这个儿子会捕捉什么？<br>Bad: 任何儿子会捕捉什么？ |
| 觉得 → 主张 | renhe_non_factive_verb | 44 | 0.8182 | 0.7273 | +0.0909 | 0.0000 | Good: 李太太不觉得拍摄电影对员工有任何影响。<br>Bad: 李太太不主张拍摄电影对员工有任何影响。 |
| 相信 → 知道 | renhe_non_factive_verb | 34 | 1.0000 | 0.9118 | +0.0882 | 0.0000 | Good: 他不相信跨越沙漠对钢琴家有任何影响。<br>Bad: 他不知道跨越沙漠对钢琴家有任何影响。 |
| 我的 → 任何 | npi_renhe_wh_question_obj | 72 | 0.5694 | 0.6528 | -0.0833 | 0.0000 | Good: 谁期待创作我的漫画？<br>Bad: 谁期待创作任何漫画？ |
| 那个 → 任何 | npi_renhe_wh_question_subj | 37 | 0.8649 | 0.9459 | -0.0811 | 0.0000 | Good: 那个妹妹会提醒什么？<br>Bad: 任何妹妹会提醒什么？ |
| multiple edits: bad deletes 任何; 我 -> 任何人 | npi_renhe_conditional | 93 | 1.0000 | 0.9247 | +0.0753 | 0.0000 | Good: 如果有任何人驾驶火车，你就告诉我。<br>Bad: 如果有人驾驶火车，你就告诉任何人。 |
| multiple edits: bad deletes 没有; bad inserts 没有 | npi_renhe_neg_scope_subj | 300 | 0.3533 | 0.3967 | -0.0433 | 0.0000 | Good: 没有任何人把大象麻醉了。<br>Bad: 任何人没有把大象麻醉了。 |
| 希望 → 主张 | renhe_non_factive_verb | 59 | 1.0000 | 0.9661 | +0.0339 | 0.0000 | Good: 她不希望弹玻璃珠对服务员有任何影响。<br>Bad: 她不主张弹玻璃珠对服务员有任何影响。 |
| 他的 → 任何 | npi_renhe_wh_question_obj | 72 | 0.7083 | 0.6806 | +0.0278 | 0.0000 | Good: 谁想煮他的鸭？<br>Bad: 谁想煮任何鸭？ |
| multiple edits: bad deletes 任何; 他 -> 任何人 | npi_renhe_conditional | 108 | 0.5278 | 0.5185 | +0.0093 | 0.0000 | Good: 如果有任何人称赞记者，你就通知他。<br>Bad: 如果有人称赞记者，你就通知任何人。 |
| 相信 → 主张 | renhe_non_factive_verb | 67 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生不相信捕捉鱼对音乐家有任何影响。<br>Bad: 李先生不主张捕捉鱼对音乐家有任何影响。 |
| 八位 → 任何 | npi_renhe_wh_question_subj | 7 | 0.7143 | 0.7143 | +0.0000 | 0.0000 | Good: 八位老师会表扬什么？<br>Bad: 任何老师会表扬什么？ |
| 五个 → 任何 | npi_renhe_wh_question_subj | 5 | 0.2000 | 0.2000 | +0.0000 | 0.0000 | Good: 五个奴隶会偷听什么东西？<br>Bad: 任何奴隶会偷听什么东西？ |
| 四个 → 任何 | npi_renhe_wh_question_subj | 4 | 0.2500 | 0.2500 | +0.0000 | 0.0000 | Good: 四个儿子会喜欢哪些东西？<br>Bad: 任何儿子会喜欢哪些东西？ |
| 三个 → 任何 | npi_renhe_wh_question_subj | 3 | 0.3333 | 0.3333 | +0.0000 | 0.0000 | Good: 三个音乐家会批评哪些东西？<br>Bad: 任何音乐家会批评哪些东西？ |
| 六个 → 任何 | npi_renhe_wh_question_subj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 六个奴隶会维护什么？<br>Bad: 任何奴隶会维护什么？ |

## passive

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| 鼻 → 杯 | passive_body_part | 17 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 冯大哥被他们打断了鼻子。<br>Bad: 冯大哥被他们打断了杯子。 |
| 鼻 → 裤 | passive_body_part | 16 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王先生被李太太检查了鼻子。<br>Bad: 王先生被李太太检查了裤子。 |
| 鼻 → 裙 | passive_body_part | 15 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太被冯大哥包扎了鼻子。<br>Bad: 何太太被冯大哥包扎了裙子。 |
| multiple edits: bad deletes 王姨; bad inserts 王姨 | BEI_construction_b | 7 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王姨被这六个记者给批评了。<br>Bad: 被这六个记者王姨给批评了。 |
| 眼睛 → 手套 | passive_body_part | 6 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小王被你检查了眼睛。<br>Bad: 小王被你检查了手套。 |
| 鼻 → 椅 | passive_body_part | 5 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她们被我打断了鼻子。<br>Bad: 她们被我打断了椅子。 |
| 心脏 → 手套 | passive_body_part | 4 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们被赵大爷检查了心脏。<br>Bad: 他们被赵大爷检查了手套。 |
| 眼睛 → 椅子 | passive_body_part | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她被张三检查了眼睛。<br>Bad: 她被张三检查了椅子。 |
| 心脏 → 椅子 | passive_body_part | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 冯大哥被张夫人检查了心脏。<br>Bad: 冯大哥被张夫人检查了椅子。 |
| 憎恨 → 微笑 | passive_intransitive | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王大娘受到她的老师憎恨。<br>Bad: 王大娘受到她的老师微笑。 |
| 批评 → 健身 | passive_intransitive | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈遭到那个领导批评。<br>Bad: 周大妈遭到那个领导健身。 |
| 批评 → 凶猛 | passive_no_adj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们的兄弟受到你批评了。<br>Bad: 你们的兄弟受到你凶猛了。 |
| 批评 → 困惑 | passive_no_adj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那七个司机遭到吴太太批评了。<br>Bad: 那七个司机遭到吴太太困惑了。 |
| 批评 → 站立 | passive_intransitive | 2 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五被她们的哥哥批评。<br>Bad: 王五被她们的哥哥站立。 |
| 批评 → 酸甜 | passive_no_adj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个儿子被张婶批评了。<br>Bad: 那个儿子被张婶酸甜了。 |
| 拥护 → 健身 | passive_intransitive | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王姨被这位舞者拥护。<br>Bad: 王姨被这位舞者健身。 |
| 称赞 → 便宜 | passive_no_adj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那一个消费者受到你称赞了。<br>Bad: 那一个消费者受到你便宜了。 |
| 称赞 → 清淡 | passive_no_adj | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那六位上级被刘先生称赞了。<br>Bad: 那六位上级被刘先生清淡了。 |
| 责备 → 玩耍 | passive_intransitive | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 吴太太遭到小明的女儿责备。<br>Bad: 吴太太遭到小明的女儿玩耍。 |
| multiple edits: bad inserts 悲; bad deletes 害 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外四个演员遭到她们伤害了。<br>Bad: 另外四个演员遭到她们悲伤了。 |
| 伤害 → 年老 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个消费者遭到你伤害了。<br>Bad: 那个消费者遭到你年老了。 |
| 伤害 → 年轻 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王五的上级被她伤害了。<br>Bad: 王五的上级被她年轻了。 |
| 伤害 → 快乐 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王小姐的姐姐受到她们伤害了。<br>Bad: 王小姐的姐姐受到她们快乐了。 |
| 伤害 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生的朋友被何太太伤害了。<br>Bad: 李先生的朋友被何太太昂贵了。 |
| 伤害 → 激进 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个工人受到他们伤害了。<br>Bad: 那个工人受到他们激进了。 |
| 伤害 → 特殊 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷的弟弟遭到她们伤害了。<br>Bad: 赵大爷的弟弟遭到她们特殊了。 |
| 劫 → 架 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王大娘被李先生的姐姐打劫。<br>Bad: 王大娘被李先生的姐姐打架。 |
| 厌恶 → 睡觉 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生受到李太太的姐姐厌恶。<br>Bad: 刘先生受到李太太的姐姐睡觉。 |
| 厌恶 → 起飞 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小王被那个弟弟厌恶。<br>Bad: 小王被那个弟弟起飞。 |
| 厌恶 → 跳舞 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李先生遭到王五的姐妹厌恶。<br>Bad: 李先生遭到王五的姐妹跳舞。 |
| 原谅 → 健身 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐被那位工人原谅。<br>Bad: 陈大姐被那位工人健身。 |
| 呵斥 → 便宜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位老师受到李先生呵斥了。<br>Bad: 那位老师受到李先生便宜了。 |
| 呵斥 → 坐下 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥被这个消费者呵斥。<br>Bad: 杨大哥被这个消费者坐下。 |
| 呵斥 → 失望 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们的母亲受到胡大爷呵斥了。<br>Bad: 我们的母亲受到胡大爷失望了。 |
| 呵斥 → 开心 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这六个工人受到我呵斥了。<br>Bad: 这六个工人受到我开心了。 |
| 呵斥 → 欢快 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那四个钢琴家受到他呵斥了。<br>Bad: 那四个钢琴家受到他欢快了。 |
| 喜欢 → 品茶 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太受到另外八个弟弟喜欢。<br>Bad: 何太太受到另外八个弟弟品茶。 |
| 嘉奖 → 入睡 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王姨被这六个上级嘉奖。<br>Bad: 王姨被这六个上级入睡。 |
| 嘉奖 → 困惑 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 她的儿子被你嘉奖了。<br>Bad: 她的儿子被你困惑了。 |
| 嘉奖 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这七个顾客被小明嘉奖了。<br>Bad: 这七个顾客被小明昂贵了。 |
| 嘉奖 → 爬行 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 赵大爷被你们的弟弟嘉奖。<br>Bad: 赵大爷被你们的弟弟爬行。 |
| 嘉奖 → 站立 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张婶受到那个姐姐嘉奖。<br>Bad: 张婶受到那个姐姐站立。 |
| 埋怨 → 停下 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷受到这两位员工埋怨。<br>Bad: 胡大爷受到这两位员工停下。 |
| 埋怨 → 微笑 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 冯大哥被小明的兄弟埋怨。<br>Bad: 冯大哥被小明的兄弟微笑。 |
| 夸奖 → 便宜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个罪犯被她们夸奖了。<br>Bad: 那个罪犯被她们便宜了。 |
| 夸奖 → 偷听 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 冯大哥受到她的领导夸奖。<br>Bad: 冯大哥受到她的领导偷听。 |
| 夸奖 → 悲伤 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这十个工人受到你夸奖了。<br>Bad: 这十个工人受到你悲伤了。 |
| 夸奖 → 清淡 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外八位司机受到他夸奖了。<br>Bad: 另外八位司机受到他清淡了。 |
| 奖励 → 便宜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外五个罪犯受到李太太奖励了。<br>Bad: 另外五个罪犯受到李太太便宜了。 |
| 奖励 → 听课 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王姨受到另外十位司机奖励。<br>Bad: 王姨受到另外十位司机听课。 |
| 奖励 → 失望 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个老板受到我们奖励了。<br>Bad: 这个老板受到我们失望了。 |
| 奖励 → 年老 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位钢琴家受到小王奖励了。<br>Bad: 那位钢琴家受到小王年老了。 |
| 奖励 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐的母亲被郑大妈奖励了。<br>Bad: 陈大姐的母亲被郑大妈昂贵了。 |
| 奖励 → 激进 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张夫人的朋友受到她奖励了。<br>Bad: 张夫人的朋友受到她激进了。 |
| 奖励 → 灵活 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这五个朋友受到王五奖励了。<br>Bad: 这五个朋友受到王五灵活了。 |
| 奖励 → 鲜嫩 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生的上级被我们奖励了。<br>Bad: 刘先生的上级被我们鲜嫩了。 |
| 嫌弃 → 玩耍 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 吴太太被胡大爷的妈妈嫌弃。<br>Bad: 吴太太被胡大爷的妈妈玩耍。 |
| 安慰 → 优雅 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷的老板受到周大妈安慰了。<br>Bad: 赵大爷的老板受到周大妈优雅了。 |
| 安慰 → 凶猛 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个舞者受到他们安慰了。<br>Bad: 这个舞者受到他们凶猛了。 |
| 安慰 → 咸鲜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这八位服务员被何太太安慰了。<br>Bad: 这八位服务员被何太太咸鲜了。 |
| 安慰 → 失望 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位员工受到你们安慰了。<br>Bad: 这位员工受到你们失望了。 |
| 安慰 → 宁静 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这四位空姐受到我安慰了。<br>Bad: 这四位空姐受到我宁静了。 |
| 安慰 → 年轻 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们的领导被他们安慰了。<br>Bad: 她们的领导被他们年轻了。 |
| 安慰 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位学生被她安慰了。<br>Bad: 这位学生被她昂贵了。 |
| 安慰 → 狂放 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你的爸爸被你们安慰了。<br>Bad: 你的爸爸被你们狂放了。 |
| 安慰 → 站立 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小王受到那位司机安慰。<br>Bad: 小王受到那位司机站立。 |
| 安慰 → 走路 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 胡大爷受到宋女士的哥哥安慰。<br>Bad: 胡大爷受到宋女士的哥哥走路。 |
| 安慰 → 醇厚 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们的兄弟被吴太太安慰了。<br>Bad: 你们的兄弟被吴太太醇厚了。 |
| 宠爱 → 闲逛 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小王受到这六个小孩宠爱。<br>Bad: 小王受到这六个小孩闲逛。 |
| 尊重 → 唱歌 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李先生受到另外两位领导尊重。<br>Bad: 李先生受到另外两位领导唱歌。 |
| 憎恨 → 停下 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生遭到那位父亲憎恨。<br>Bad: 刘先生遭到那位父亲停下。 |
| 憎恨 → 叹息 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷遭到我的姐姐憎恨。<br>Bad: 赵大爷遭到我的姐姐叹息。 |
| 憎恨 → 闲逛 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 赵大爷遭到冯大哥的姐姐憎恨。<br>Bad: 赵大爷遭到冯大哥的姐姐闲逛。 |
| 打劫 → 冷静 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位下属遭到张三打劫了。<br>Bad: 这位下属遭到张三冷静了。 |
| 批判 → 爬行 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小明遭到那八个司机批判。<br>Bad: 小明遭到那八个司机爬行。 |
| 批判 → 走路 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 赵大爷受到这位工人批判。<br>Bad: 赵大爷受到这位工人走路。 |
| 批评 → 启程 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 胡大爷受到我们的妈妈批评。<br>Bad: 胡大爷受到我们的妈妈启程。 |
| 批评 → 宁静 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们的姐妹遭到我们批评了。<br>Bad: 我们的姐妹遭到我们宁静了。 |
| 批评 → 激昂 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外六位顾客被你们批评了。<br>Bad: 另外六位顾客被你们激昂了。 |
| 抨击 → 睡觉 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 何太太被那个下属抨击。<br>Bad: 何太太被那个下属睡觉。 |
| 拥护 → 呼吸 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 何太太受到那个司机拥护。<br>Bad: 何太太受到那个司机呼吸。 |
| 拥护 → 爬行 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 赵大爷受到她的老板拥护。<br>Bad: 赵大爷受到她的老板爬行。 |
| 排挤 → 叹息 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李四受到那个领导排挤。<br>Bad: 李四受到那个领导叹息。 |
| 推崇 → 偷听 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张三受到那个小孩推崇。<br>Bad: 张三受到那个小孩偷听。 |
| 推崇 → 入睡 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张婶被这两个服务员推崇。<br>Bad: 张婶被这两个服务员入睡。 |
| 推崇 → 闲逛 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 刘先生受到赵大爷的女儿推崇。<br>Bad: 刘先生受到赵大爷的女儿闲逛。 |
| 提醒 → 伤心 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位打工人受到他们提醒了。<br>Bad: 那位打工人受到他们伤心了。 |
| 提醒 → 宁静 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李太太的老板受到王姨提醒了。<br>Bad: 李太太的老板受到王姨宁静了。 |
| 提醒 → 愤怒 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们的老板被他提醒了。<br>Bad: 我们的老板被他愤怒了。 |
| 提醒 → 清淡 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外九个弟弟受到他们提醒了。<br>Bad: 另外九个弟弟受到他们清淡了。 |
| 提醒 → 舒缓 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位父亲遭到冯大哥提醒了。<br>Bad: 这位父亲遭到冯大哥舒缓了。 |
| 支持 → 健身 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈被那七个姐姐支持。<br>Bad: 周大妈被那七个姐姐健身。 |
| 支持 → 微笑 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生受到刘先生的爸爸支持。<br>Bad: 王先生受到刘先生的爸爸微笑。 |
| 教育 → 坎坷 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这七位舞者受到你们教育了。<br>Bad: 这七位舞者受到你们坎坷了。 |
| 教育 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐的老师遭到宋女士教育了。<br>Bad: 陈大姐的老师遭到宋女士昂贵了。 |
| 教育 → 欢快 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 他们的姐姐遭到王先生教育了。<br>Bad: 他们的姐姐遭到王先生欢快了。 |
| 教育 → 激进 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那八位员工遭到我们教育了。<br>Bad: 那八位员工遭到我们激进了。 |
| 教育 → 特殊 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位吉他手被他们教育了。<br>Bad: 那位吉他手被他们特殊了。 |
| 教育 → 睡觉 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 徐小姐受到那个小孩教育。<br>Bad: 徐小姐受到那个小孩睡觉。 |
| 教育 → 过来 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王大娘遭到那个舞者教育。<br>Bad: 王大娘遭到那个舞者过来。 |
| 欺骗 → 听课 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张婶受到那位下属欺骗。<br>Bad: 张婶受到那位下属听课。 |
| 欺骗 → 坐下 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 胡大爷被另外两位母亲欺骗。<br>Bad: 胡大爷被另外两位母亲坐下。 |
| 欺骗 → 走路 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 杨大哥遭到那位员工欺骗。<br>Bad: 杨大哥遭到那位员工走路。 |
| 照顾 → 便宜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这三个吉他手受到刘先生照顾了。<br>Bad: 这三个吉他手受到刘先生便宜了。 |
| 照顾 → 凶猛 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这九位顾客被你照顾了。<br>Bad: 这九位顾客被你凶猛了。 |
| 照顾 → 困惑 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个哥哥被刘先生照顾了。<br>Bad: 那个哥哥被刘先生困惑了。 |
| 照顾 → 悠扬 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这两位音乐家受到冯大哥照顾了。<br>Bad: 这两位音乐家受到冯大哥悠扬了。 |
| 照顾 → 温驯 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那十位空姐受到李太太照顾了。<br>Bad: 那十位空姐受到李太太温驯了。 |
| 照顾 → 甘甜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个奴隶被王姨照顾了。<br>Bad: 这个奴隶被王姨甘甜了。 |
| 照顾 → 躺下 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太受到那个下属照顾。<br>Bad: 何太太受到那个下属躺下。 |
| 爱戴 → 打架 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥被这位打工人爱戴。<br>Bad: 杨大哥被这位打工人打架。 |
| 爱戴 → 玩耍 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王小姐受到你们的儿子爱戴。<br>Bad: 王小姐受到你们的儿子玩耍。 |
| 眼睛 → 被子 | passive_body_part | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李四被何太太检查了眼睛。<br>Bad: 李四被何太太检查了被子。 |
| 称赞 → 低沉 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你的哥哥被杨大哥称赞了。<br>Bad: 你的哥哥被杨大哥低沉了。 |
| 称赞 → 悲伤 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位同事受到何太太称赞了。<br>Bad: 那位同事受到何太太悲伤了。 |
| 称赞 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个司机被她们称赞了。<br>Bad: 那个司机被她们昂贵了。 |
| 称赞 → 激进 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位空姐被他们称赞了。<br>Bad: 这位空姐被他们激进了。 |
| 称赞 → 灵活 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位母亲被你们称赞了。<br>Bad: 那位母亲被你们灵活了。 |
| 称赞 → 看戏 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太受到胡大爷的老板称赞。<br>Bad: 何太太受到胡大爷的老板看戏。 |
| 称赞 → 粗旷 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个老板被何太太称赞了。<br>Bad: 那个老板被何太太粗旷了。 |
| 称赞 → 舒缓 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这个哥哥受到张先生称赞了。<br>Bad: 这个哥哥受到张先生舒缓了。 |
| 约束 → 便宜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个演员遭到周大妈约束了。<br>Bad: 那个演员遭到周大妈便宜了。 |
| 约束 → 保守 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位工人被我约束了。<br>Bad: 这位工人被我保守了。 |
| 约束 → 温驯 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们的妈妈被徐小姐约束了。<br>Bad: 我们的妈妈被徐小姐温驯了。 |
| 约束 → 激进 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们的姐姐遭到我约束了。<br>Bad: 她们的姐姐遭到我激进了。 |
| 约束 → 热烈 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这位演员被我们约束了。<br>Bad: 这位演员被我们热烈了。 |
| 约束 → 爬行 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 王五受到那位空姐约束。<br>Bad: 王五受到那位空姐爬行。 |
| 约束 → 甘甜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 周大妈的老板被小明约束了。<br>Bad: 周大妈的老板被小明甘甜了。 |
| 维护 → 健身 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张夫人被另外三个女儿维护。<br>Bad: 张夫人被另外三个女儿健身。 |
| 维护 → 出发 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥受到你的员工维护。<br>Bad: 杨大哥受到你的员工出发。 |
| 耳朵 → 裤子 | passive_body_part | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 何太太被我们包扎了耳朵。<br>Bad: 何太太被我们包扎了裤子。 |
| 肚 → 桌 | passive_body_part | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 张婶被胡大爷包扎了肚子。<br>Bad: 张婶被胡大爷包扎了桌子。 |
| 表扬 → 凶猛 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个儿子受到他们表扬了。<br>Bad: 这个儿子受到他们凶猛了。 |
| 表扬 → 叹息 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 杨大哥被另外六个舞者表扬。<br>Bad: 杨大哥被另外六个舞者叹息。 |
| 表扬 → 狂放 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 另外六个消费者被赵大爷表扬了。<br>Bad: 另外六个消费者被赵大爷狂放了。 |
| 表扬 → 苦恼 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这个打工人受到他表扬了。<br>Bad: 这个打工人受到他苦恼了。 |
| 诽谤 → 坎坷 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位同事受到我们诽谤了。<br>Bad: 这位同事受到我们坎坷了。 |
| 诽谤 → 看戏 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 徐小姐受到那位上级诽谤。<br>Bad: 徐小姐受到那位上级看戏。 |
| 诽谤 → 舒缓 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这两个下属受到她们诽谤了。<br>Bad: 这两个下属受到她们舒缓了。 |
| 责备 → 入睡 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 郑大妈被另外一位老师责备。<br>Bad: 郑大妈被另外一位老师入睡。 |
| 责备 → 开心 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我的姐妹被王五责备了。<br>Bad: 我的姐妹被王五开心了。 |
| 责备 → 欢快 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位记者受到你们责备了。<br>Bad: 那位记者受到你们欢快了。 |
| 责备 → 热烈 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外六位母亲遭到他责备了。<br>Bad: 另外六位母亲遭到他热烈了。 |
| 责备 → 狂野 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我的老板受到她责备了。<br>Bad: 我的老板受到她狂野了。 |
| 责备 → 苦恼 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那位钢琴家遭到她们责备了。<br>Bad: 那位钢琴家遭到她们苦恼了。 |
| 赞成 → 健身 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐被这位吉他手赞成。<br>Bad: 陈大姐被这位吉他手健身。 |
| 赞成 → 出发 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 郑大妈被那位顾客赞成。<br>Bad: 郑大妈被那位顾客出发。 |
| 赞成 → 微笑 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王先生被小王的哥哥赞成。<br>Bad: 王先生被小王的哥哥微笑。 |
| 赞成 → 睡觉 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 刘先生被赵大爷的姐妹赞成。<br>Bad: 刘先生被赵大爷的姐妹睡觉。 |
| 辩护 → 微笑 | passive_intransitive | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 李四受到那位下属辩护。<br>Bad: 李四受到那位下属微笑。 |
| 鼓励 → 优雅 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这三位钢琴家被我们鼓励了。<br>Bad: 这三位钢琴家被我们优雅了。 |
| 鼓励 → 便宜 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 这六个小孩被陈大姐鼓励了。<br>Bad: 这六个小孩被陈大姐便宜了。 |
| 鼓励 → 凶猛 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位服务员受到李四鼓励了。<br>Bad: 这位服务员受到李四凶猛了。 |
| 鼓励 → 悠扬 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李先生的姐妹被她们鼓励了。<br>Bad: 李先生的姐妹被她们悠扬了。 |
| 鼓励 → 昂贵 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 那个奴隶被我们鼓励了。<br>Bad: 那个奴隶被我们昂贵了。 |
| 鼓励 → 普通 | passive_no_adj | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 另外八位记者被你鼓励了。<br>Bad: 另外八位记者被你普通了。 |
| 鼓励 → 站立 | passive_intransitive | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张先生受到这三个顾客鼓励。<br>Bad: 张先生受到这三个顾客站立。 |
| 鼓励 → 苦恼 | passive_no_adj | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 这位母亲受到冯大哥鼓励了。<br>Bad: 这位母亲受到冯大哥苦恼了。 |
| bad deletes 小明 | passive_agent_deletion_short | 8 | 1.0000 | 0.1250 | +0.8750 | 0.0000 | Good: 胡大爷被小明在欧盟控制了。<br>Bad: 胡大爷被在欧盟控制了。 |
| 耳朵 → 裙子 | passive_body_part | 8 | 0.0000 | 0.8750 | -0.8750 | 0.0000 | Good: 胡大爷被她们包扎了耳朵。<br>Bad: 胡大爷被她们包扎了裙子。 |
| 鼻 → 袜 | passive_body_part | 23 | 0.0000 | 0.8696 | -0.8696 | 0.0000 | Good: 小王被我们检查了鼻子。<br>Bad: 小王被我们检查了袜子。 |
| bad deletes 她 | passive_agent_deletion_long_right_b | 16 | 0.8750 | 0.0625 | +0.8125 | 0.0000 | Good: 小明被她派王姨嘉奖了。<br>Bad: 小明被派王姨嘉奖了。 |
| bad deletes 张先生 | passive_suo | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 这个新闻不可以被张先生所知晓。<br>Bad: 这个新闻不可以被所知晓。 |
| bad deletes 李先生 | passive_suo | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 那个秘密不可以被李先生所知晓。<br>Bad: 那个秘密不可以被所知晓。 |
| multiple edits: bad deletes 李先生; bad inserts 李先生 | BEI_construction_b | 7 | 1.0000 | 0.2857 | +0.7143 | 0.0000 | Good: 李先生被另外七位下属给呵斥了。<br>Bad: 被另外七位下属李先生给呵斥了。 |
| bad deletes 小明 | passive_suo | 10 | 1.0000 | 0.3000 | +0.7000 | 0.0000 | Good: 这个事不应该被小明所了解。<br>Bad: 这个事不应该被所了解。 |
| bad deletes 他 | passive_agent_deletion_long_right_b | 16 | 0.9375 | 0.2500 | +0.6875 | 0.0000 | Good: 徐小姐被他派吴太太奖励了。<br>Bad: 徐小姐被派吴太太奖励了。 |
| multiple edits: bad deletes 赵大爷; bad inserts 赵大爷 | BEI_construction_b | 9 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 赵大爷被那位老师给奖励了。<br>Bad: 被那位老师赵大爷给奖励了。 |
| bad deletes 小明 | passive_agent_deletion_long_right_b | 6 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 张夫人被小明派王小姐表扬了。<br>Bad: 张夫人被派王小姐表扬了。 |
| multiple edits: bad deletes 王小姐; bad inserts 王小姐 | BEI_construction_b | 6 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 王小姐被这个朋友给表扬了。<br>Bad: 被这个朋友王小姐给表扬了。 |
| bad deletes 张婶 | passive_suo | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 这些丑闻不应该被张婶所知晓。<br>Bad: 这些丑闻不应该被所知晓。 |
| bad deletes 徐小姐 | passive_suo | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 这些事情不可以被徐小姐所知晓。<br>Bad: 这些事情不可以被所知晓。 |
| 批评 → 冷静 | passive_no_adj | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 这位同事被张先生批评了。<br>Bad: 这位同事被张先生冷静了。 |
| 鼻 → 桌 | passive_body_part | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 郑大妈被王大娘包扎了鼻子。<br>Bad: 郑大妈被王大娘包扎了桌子。 |
| multiple edits: bad deletes 飞机上被; bad inserts 飞机上 | BEI_deletion | 29 | 0.3103 | 0.9655 | -0.6552 | 0.0000 | Good: 飞机上被李先生扔满了电冰箱。<br>Bad: 李先生扔满了飞机上电冰箱。 |
| multiple edits: bad deletes 胡大爷; bad inserts 胡大爷 | BEI_construction_b | 8 | 1.0000 | 0.3750 | +0.6250 | 0.0000 | Good: 胡大爷被另外六位领导给欺负了。<br>Bad: 被另外六位领导胡大爷给欺负了。 |
| bad deletes 王大娘 | passive_suo | 10 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 那些信息不可以被王大娘所了解。<br>Bad: 那些信息不可以被所了解。 |
| 肚子 → 衣服 | passive_body_part | 10 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 冯大哥被你包扎了肚子。<br>Bad: 冯大哥被你包扎了衣服。 |
| bad deletes 吴太太 | passive_suo | 5 | 0.4000 | 1.0000 | -0.6000 | 0.0000 | Good: 这个秘密不应该被吴太太所知晓。<br>Bad: 这个秘密不应该被所知晓。 |
| 肚子 → 手套 | passive_body_part | 12 | 1.0000 | 0.4167 | +0.5833 | 0.0000 | Good: 我被他包扎了肚子。<br>Bad: 我被他包扎了手套。 |
| bad deletes 她们 | passive_agent_deletion_long_right_b | 21 | 0.0000 | 0.5714 | -0.5714 | 0.0000 | Good: 郑大妈被她们派张先生原谅了。<br>Bad: 郑大妈被派张先生原谅了。 |
| bad deletes 王小姐 | passive_suo | 7 | 0.8571 | 0.2857 | +0.5714 | 0.0000 | Good: 那些秘密不应该被王小姐所知晓。<br>Bad: 那些秘密不应该被所知晓。 |
| multiple edits: bad deletes 李太太; bad inserts 李太太 | BEI_construction_b | 7 | 1.0000 | 0.4286 | +0.5714 | 0.0000 | Good: 李太太被这九个姐姐给欺骗了。<br>Bad: 被这九个姐姐李太太给欺骗了。 |
| multiple edits: bad inserts 被他; bad deletes 被他 | BEI_construction_a | 25 | 0.8800 | 0.3200 | +0.5600 | 0.0000 | Good: 张夫人的这八块糖果被他给买了。<br>Bad: 被他张夫人的这八块糖果给买了。 |
| 耳朵 → 手套 | passive_body_part | 15 | 1.0000 | 0.4667 | +0.5333 | 0.0000 | Good: 赵大爷被王小姐检查了耳朵。<br>Bad: 赵大爷被王小姐检查了手套。 |
| bad deletes 你们 | passive_agent_deletion_short | 17 | 0.8824 | 0.3529 | +0.5294 | 0.0000 | Good: 王先生被你们在联合国伤害了。<br>Bad: 王先生被在联合国伤害了。 |
| 耳朵 → 杯子 | passive_body_part | 10 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 李太太被刘先生检查了耳朵。<br>Bad: 李太太被刘先生检查了杯子。 |
| bad deletes 何太太 | passive_suo | 6 | 0.8333 | 0.3333 | +0.5000 | 0.0000 | Good: 这些信息不应该被何太太所知晓。<br>Bad: 这些信息不应该被所知晓。 |
| multiple edits: bad inserts 被陈大姐; bad deletes 被陈大姐 | BEI_construction_a | 6 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你的这只鼻子被陈大姐给包扎了。<br>Bad: 被陈大姐你的这只鼻子给包扎了。 |
| bad deletes 赵大爷 | passive_suo | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这个丑闻不应该被赵大爷所了解。<br>Bad: 这个丑闻不应该被所了解。 |
| bad deletes 郑大妈 | passive_suo | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这个东西不可以被郑大妈所知道。<br>Bad: 这个东西不可以被所知道。 |
| multiple edits: bad inserts 被冯大哥; bad deletes 被冯大哥 | BEI_construction_a | 4 | 0.7500 | 0.2500 | +0.5000 | 0.0000 | Good: 王大娘的这三张桌子被冯大哥给搬了。<br>Bad: 被冯大哥王大娘的这三张桌子给搬了。 |
| multiple edits: bad inserts 被李先生; bad deletes 被李先生 | BEI_construction_a | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 周大妈的那条鱼被李先生给爆炒了。<br>Bad: 被李先生周大妈的那条鱼给爆炒了。 |
| 鼻 → 被 | passive_body_part | 4 | 0.2500 | 0.7500 | -0.5000 | 0.0000 | Good: 我们被何太太检查了鼻子。<br>Bad: 我们被何太太检查了被子。 |
| bad deletes 刘先生 | passive_agent_deletion_long_right_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 小明被刘先生派杨大哥责备了。<br>Bad: 小明被派杨大哥责备了。 |
| bad deletes 宋女士 | passive_agent_deletion_long_right_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 郑大妈被宋女士派何太太批评了。<br>Bad: 郑大妈被派何太太批评了。 |
| bad deletes 张夫人 | passive_suo | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这些新闻不应该被张夫人所知道。<br>Bad: 这些新闻不应该被所知道。 |
| bad deletes 李太太 | passive_agent_deletion_long_right_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 周大妈被李太太派陈大姐呵斥了。<br>Bad: 周大妈被派陈大姐呵斥了。 |
| bad deletes 赵大爷 | passive_agent_deletion_short | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 何太太被赵大爷在欧盟表扬了。<br>Bad: 何太太被在欧盟表扬了。 |
| 厌恶 → 品茶 | passive_intransitive | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 小明受到张夫人的朋友厌恶。<br>Bad: 小明受到张夫人的朋友品茶。 |
| 原谅 → 颤抖 | passive_intransitive | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 刘先生被另外五个吉他手原谅。<br>Bad: 刘先生被另外五个吉他手颤抖。 |
| 嘉奖 → 悲伤 | passive_no_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 另外七个服务员被我们嘉奖了。<br>Bad: 另外七个服务员被我们悲伤了。 |
| 嘉奖 → 酥脆 | passive_no_adj | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 另外五位顾客被我嘉奖了。<br>Bad: 另外五位顾客被我酥脆了。 |
| 安慰 → 无聊 | passive_no_adj | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 那三个领导被王大娘安慰了。<br>Bad: 那三个领导被王大娘无聊了。 |
| 心脏 → 裙子 | passive_body_part | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 陈大姐被吴太太检查了心脏。<br>Bad: 陈大姐被吴太太检查了裙子。 |
| 憎恨 → 颤抖 | passive_intransitive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 杨大哥被她的爸爸憎恨。<br>Bad: 杨大哥被她的爸爸颤抖。 |
| 批评 → 躺下 | passive_intransitive | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 王五遭到我的老师批评。<br>Bad: 王五遭到我的老师躺下。 |
| 支持 → 睡觉 | passive_intransitive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 小王被那位工人支持。<br>Bad: 小王被那位工人睡觉。 |
| 照顾 → 优雅 | passive_no_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这两个上级被李太太照顾了。<br>Bad: 这两个上级被李太太优雅了。 |
| 爱戴 → 站立 | passive_intransitive | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 冯大哥被周大妈的下属爱戴。<br>Bad: 冯大哥被周大妈的下属站立。 |
| 称赞 → 玩耍 | passive_intransitive | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 王小姐被王姨的上级称赞。<br>Bad: 王小姐被王姨的上级玩耍。 |
| 约束 → 酥软 | passive_no_adj | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那三个吉他手遭到王先生约束了。<br>Bad: 那三个吉他手遭到王先生酥软了。 |
| 维护 → 启程 | passive_intransitive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 王先生被那个弟弟维护。<br>Bad: 王先生被那个弟弟启程。 |
| 肚 → 椅 | passive_body_part | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 我们被你们包扎了肚子。<br>Bad: 我们被你们包扎了椅子。 |
| 肚 → 被 | passive_body_part | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 她被小明检查了肚子。<br>Bad: 她被小明检查了被子。 |
| 赞赏 → 睡觉 | passive_intransitive | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 小王被那位同事赞赏。<br>Bad: 小王被那位同事睡觉。 |
| bad deletes 李先生 | passive_agent_deletion_short | 11 | 1.0000 | 0.5455 | +0.4545 | 0.0000 | Good: 胡大爷被李先生在非洲联盟鼓励了。<br>Bad: 胡大爷被在非洲联盟鼓励了。 |
| multiple edits: bad deletes 货箱上被; bad inserts 货箱上 | BEI_deletion | 126 | 0.3413 | 0.7937 | -0.4524 | 0.0000 | Good: 货箱上被王大娘藏满了手套。<br>Bad: 王大娘藏满了货箱上手套。 |
| bad deletes 王姨 | passive_suo | 9 | 0.8889 | 0.4444 | +0.4444 | 0.0000 | Good: 这个消息不应该被王姨所知道。<br>Bad: 这个消息不应该被所知道。 |
| bad deletes 何太太 | passive_agent_deletion_long_right_b | 7 | 1.0000 | 0.5714 | +0.4286 | 0.0000 | Good: 冯大哥被何太太派王五提醒了。<br>Bad: 冯大哥被派王五提醒了。 |
| bad deletes 李四 | passive_agent_deletion_short | 7 | 1.0000 | 0.5714 | +0.4286 | 0.0000 | Good: 张婶被李四在联合国提醒了。<br>Bad: 张婶被在联合国提醒了。 |
| multiple edits: bad inserts 被何太太; bad deletes 被何太太 | BEI_construction_a | 7 | 1.0000 | 0.5714 | +0.4286 | 0.0000 | Good: 张三的那张桌子被何太太给搬了。<br>Bad: 被何太太张三的那张桌子给搬了。 |
| bad deletes 我 | passive_agent_deletion_short | 17 | 0.2941 | 0.7059 | -0.4118 | 0.0000 | Good: 陈大姐被我在奥委会诽谤了。<br>Bad: 陈大姐被在奥委会诽谤了。 |
| bad deletes 他 | passive_suo | 22 | 0.9091 | 0.5000 | +0.4091 | 0.0000 | Good: 这个秘密不应该被他所知道。<br>Bad: 这个秘密不应该被所知道。 |
| multiple edits: bad deletes 轮船上被; bad inserts 轮船上 | BEI_deletion | 37 | 1.0000 | 0.5946 | +0.4054 | 0.0000 | Good: 轮船上被徐小姐放满了手套。<br>Bad: 徐小姐放满了轮船上手套。 |
| bad deletes 宋女士 | passive_agent_deletion_short | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 李太太被宋女士在非洲联盟诽谤了。<br>Bad: 李太太被在非洲联盟诽谤了。 |
| bad deletes 李先生 | passive_agent_deletion_long_right_b | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 张婶被李先生派王先生欺骗了。<br>Bad: 张婶被派王先生欺骗了。 |
| bad deletes 李四 | passive_agent_deletion_long_right_b | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 郑大妈被李四派徐小姐约束了。<br>Bad: 郑大妈被派徐小姐约束了。 |
| multiple edits: bad deletes 张三; bad inserts 张三 | BEI_construction_b | 5 | 0.8000 | 0.4000 | +0.4000 | 0.0000 | Good: 张三被她们的下属给打劫了。<br>Bad: 被她们的下属张三给打劫了。 |
| multiple edits: bad inserts 被李四; bad deletes 被李四 | BEI_construction_a | 5 | 0.8000 | 0.4000 | +0.4000 | 0.0000 | Good: 徐小姐的另外一杯橙汁被李四给买了。<br>Bad: 被李四徐小姐的另外一杯橙汁给买了。 |
| multiple edits: bad deletes 周大妈; bad inserts 周大妈 | BEI_construction_b | 5 | 0.6000 | 0.2000 | +0.4000 | 0.0000 | Good: 周大妈被那一位演奏员给称赞了。<br>Bad: 被那一位演奏员周大妈给称赞了。 |
| bad deletes 我们 | passive_agent_deletion_long_right_b | 21 | 0.0000 | 0.3810 | -0.3810 | 0.0000 | Good: 冯大哥被我们派徐小姐夸奖了。<br>Bad: 冯大哥被派徐小姐夸奖了。 |
| multiple edits: bad deletes 她; bad inserts 她 | BEI_preposition | 37 | 0.8919 | 0.5135 | +0.3784 | 0.0000 | Good: 她被张三批评了。<br>Bad: 被张三她批评了。 |
| bad deletes 周大妈 | passive_suo | 8 | 1.0000 | 0.6250 | +0.3750 | 0.0000 | Good: 这些新闻不可以被周大妈所了解。<br>Bad: 这些新闻不可以被所了解。 |
| bad deletes 她 | passive_suo | 22 | 1.0000 | 0.6364 | +0.3636 | 0.0000 | Good: 这个东西不应该被她所了解。<br>Bad: 这个东西不应该被所了解。 |
| bad deletes 张三 | passive_suo | 11 | 0.4545 | 0.0909 | +0.3636 | 0.0000 | Good: 这些信息不应该被张三所知道。<br>Bad: 这些信息不应该被所知道。 |
| multiple edits: bad inserts 被你们; bad deletes 被你们 | BEI_construction_a | 25 | 0.8800 | 0.5200 | +0.3600 | 0.0000 | Good: 我们的那四头大象被你们给麻醉了。<br>Bad: 被你们我们的那四头大象给麻醉了。 |
| bad deletes 刘先生 | passive_agent_deletion_short | 9 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 陈大姐被刘先生在北约提醒了。<br>Bad: 陈大姐被在北约提醒了。 |
| bad deletes 张婶 | passive_agent_deletion_long_right_b | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 王姨被张婶派胡大爷安慰了。<br>Bad: 王姨被派胡大爷安慰了。 |
| bad deletes 陈大姐 | passive_suo | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这个东西不可以被陈大姐所知道。<br>Bad: 这个东西不可以被所知道。 |
| multiple edits: bad deletes 何太太; bad inserts 何太太 | BEI_construction_b | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 何太太被他的爸爸给批评了。<br>Bad: 被他的爸爸何太太给批评了。 |
| multiple edits: bad deletes 张夫人; bad inserts 张夫人 | BEI_construction_b | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 张夫人被你们的儿子给呵斥了。<br>Bad: 被你们的儿子张夫人给呵斥了。 |
| multiple edits: bad deletes 郑大妈; bad inserts 郑大妈 | BEI_construction_b | 6 | 0.5000 | 0.8333 | -0.3333 | 0.0000 | Good: 郑大妈被这七位员工给表扬了。<br>Bad: 被这七位员工郑大妈给表扬了。 |
| bad deletes 胡大爷 | passive_suo | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这些秘密不可以被胡大爷所知道。<br>Bad: 这些秘密不可以被所知道。 |
| multiple edits: bad deletes 王先生; bad inserts 王先生 | BEI_construction_b | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 王先生被王小姐的女儿给约束了。<br>Bad: 被王小姐的女儿王先生给约束了。 |
| multiple edits: bad inserts 被张婶; bad deletes 被张婶 | BEI_construction_a | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 那只鸡被张婶给清蒸了。<br>Bad: 被张婶那只鸡给清蒸了。 |
| 表扬 → 激进 | passive_no_adj | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 那位空姐被李四表扬了。<br>Bad: 那位空姐被李四激进了。 |
| bad deletes 宋女士 | passive_suo | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 那些事不可以被宋女士所知道。<br>Bad: 那些事不可以被所知道。 |
| bad deletes 李四 | passive_suo | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 这个新闻不应该被李四所知道。<br>Bad: 这个新闻不应该被所知道。 |
| 批评 → 品茶 | passive_intransitive | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 宋女士受到她们的母亲批评。<br>Bad: 宋女士受到她们的母亲品茶。 |
| 批评 → 走路 | passive_intransitive | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 张先生被这位吉他手批评。<br>Bad: 张先生被这位吉他手走路。 |
| 眼睛 → 杯子 | passive_body_part | 3 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 她们被他们检查了眼睛。<br>Bad: 她们被他们检查了杯子。 |
| multiple edits: bad deletes 刘先生; bad inserts 刘先生 | BEI_construction_b | 9 | 0.8889 | 0.5556 | +0.3333 | 0.0000 | Good: 刘先生被这两位吉他手给夸奖了。<br>Bad: 被这两位吉他手刘先生给夸奖了。 |
| bad deletes 她们 | passive_suo | 19 | 0.6316 | 0.9474 | -0.3158 | 0.0000 | Good: 那个新闻不应该被她们所知晓。<br>Bad: 那个新闻不应该被所知晓。 |
| multiple edits: bad inserts 被你; bad deletes 被你 | BEI_construction_a | 19 | 1.0000 | 0.6842 | +0.3158 | 0.0000 | Good: 王姨的那个玻璃珠被你给弹了。<br>Bad: 被你王姨的那个玻璃珠给弹了。 |
| bad deletes 她们 | passive_agent_deletion_short | 28 | 0.5714 | 0.2857 | +0.2857 | 0.0000 | Good: 王小姐被她们在北约夸奖了。<br>Bad: 王小姐被在北约夸奖了。 |
| 肚 → 袜 | passive_body_part | 7 | 1.0000 | 0.7143 | +0.2857 | 0.0000 | Good: 你被我们检查了肚子。<br>Bad: 你被我们检查了袜子。 |
| multiple edits: bad deletes 货车上被; bad inserts 货车上 | BEI_deletion | 39 | 0.6667 | 0.3846 | +0.2821 | 0.0000 | Good: 货车上被张夫人放满了电冰箱。<br>Bad: 张夫人放满了货车上电冰箱。 |
| multiple edits: bad deletes 他们; bad inserts 他们 | BEI_preposition | 32 | 0.6875 | 0.9375 | -0.2500 | 0.0000 | Good: 他们被杨大哥诽谤了。<br>Bad: 被杨大哥他们诽谤了。 |
| multiple edits: bad deletes 他; bad inserts 他 | BEI_construction_b | 20 | 0.5500 | 0.8000 | -0.2500 | 0.0000 | Good: 他被另外一个工人给打劫了。<br>Bad: 被另外一个工人他给打劫了。 |
| bad deletes 张夫人 | passive_agent_deletion_short | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 周大妈被张夫人在非洲联盟伤害了。<br>Bad: 周大妈被在非洲联盟伤害了。 |
| multiple edits: bad inserts 被李太太; bad deletes 被李太太 | BEI_construction_a | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 那三片面包被李太太给买了。<br>Bad: 被李太太那三片面包给买了。 |
| bad deletes 刘先生 | passive_suo | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 这些事不应该被刘先生所知道。<br>Bad: 这些事不应该被所知道。 |
| bad deletes 吴太太 | passive_agent_deletion_long_right_b | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 张先生被吴太太派张三夸奖了。<br>Bad: 张先生被派张三夸奖了。 |
| bad deletes 王先生 | passive_suo | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 这些东西不可以被王先生所知晓。<br>Bad: 这些东西不可以被所知晓。 |
| bad deletes 王姨 | passive_agent_deletion_short | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 胡大爷被王姨在欧盟夸奖了。<br>Bad: 胡大爷被在欧盟夸奖了。 |
| multiple edits: bad deletes 陈大姐; bad inserts 陈大姐 | BEI_construction_b | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 陈大姐被那四位舞者给约束了。<br>Bad: 被那四位舞者陈大姐给约束了。 |
| multiple edits: bad inserts 被王大娘; bad deletes 被王大娘 | BEI_construction_a | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 徐小姐的那杯白酒被王大娘给买了。<br>Bad: 被王大娘徐小姐的那杯白酒给买了。 |
| multiple edits: bad deletes 王大娘; bad inserts 王大娘 | BEI_construction_b | 9 | 0.6667 | 0.8889 | -0.2222 | 0.0000 | Good: 王大娘被那个音乐家给鼓励了。<br>Bad: 被那个音乐家王大娘给鼓励了。 |
| multiple edits: bad inserts 被他们; bad deletes 被他们 | BEI_construction_a | 14 | 0.7143 | 0.9286 | -0.2143 | 0.0000 | Good: 那两桶啤酒被他们给买了。<br>Bad: 被他们那两桶啤酒给买了。 |
| multiple edits: bad inserts 被她; bad deletes 被她 | BEI_construction_a | 24 | 0.7500 | 0.5417 | +0.2083 | 0.0000 | Good: 张先生的另外三桶方便面被她给买了。<br>Bad: 被她张先生的另外三桶方便面给买了。 |
| multiple edits: bad inserts 被张三; bad deletes 被张三 | BEI_construction_a | 5 | 0.6000 | 0.8000 | -0.2000 | 0.0000 | Good: 他的那九桶啤酒被张三给喝了。<br>Bad: 被张三他的那九桶啤酒给喝了。 |
| bad deletes 王五 | passive_suo | 5 | 0.0000 | 0.2000 | -0.2000 | 0.0000 | Good: 这个秘密不可以被王五所知晓。<br>Bad: 这个秘密不可以被所知晓。 |
| 眼睛 → 裙子 | passive_body_part | 5 | 0.0000 | 0.2000 | -0.2000 | 0.0000 | Good: 她们被冯大哥检查了眼睛。<br>Bad: 她们被冯大哥检查了裙子。 |
| bad deletes 小王 | passive_suo | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 这个新闻不可以被小王所知道。<br>Bad: 这个新闻不可以被所知道。 |
| bad deletes 王五 | passive_agent_deletion_long_right_b | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 何太太被王五派徐小姐欺负了。<br>Bad: 何太太被派徐小姐欺负了。 |
| multiple edits: bad inserts 被徐小姐; bad deletes 被徐小姐 | BEI_construction_a | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 另外十张桌子被徐小姐给搬了。<br>Bad: 被徐小姐另外十张桌子给搬了。 |
| multiple edits: bad inserts 被王先生; bad deletes 被王先生 | BEI_construction_a | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 那头大象被王先生给麻醉了。<br>Bad: 被王先生那头大象给麻醉了。 |
| multiple edits: bad inserts 被王姨; bad deletes 被王姨 | BEI_construction_a | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 那七桶方便面被王姨给吃了。<br>Bad: 被王姨那七桶方便面给吃了。 |
| bad deletes 张夫人 | passive_agent_deletion_long_right_b | 11 | 1.0000 | 0.8182 | +0.1818 | 0.0000 | Good: 何太太被张夫人派小王表扬了。<br>Bad: 何太太被派小王表扬了。 |
| bad deletes 他们 | passive_agent_deletion_long_right_b | 18 | 0.3889 | 0.5556 | -0.1667 | 0.0000 | Good: 胡大爷被他们派王小姐欺骗了。<br>Bad: 胡大爷被派王小姐欺骗了。 |
| multiple edits: bad inserts 被王五; bad deletes 被王五 | BEI_construction_a | 6 | 0.5000 | 0.3333 | +0.1667 | 0.0000 | Good: 我的另外五头大象被王五给麻醉了。<br>Bad: 被王五我的另外五头大象给麻醉了。 |
| bad deletes 你 | passive_agent_deletion_short | 18 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 王大娘被你在联合国嘉奖了。<br>Bad: 王大娘被在联合国嘉奖了。 |
| bad deletes 徐小姐 | passive_agent_deletion_short | 6 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 张三被徐小姐在世界银行奖励了。<br>Bad: 张三被在世界银行奖励了。 |
| bad deletes 王五 | passive_agent_deletion_short | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 胡大爷被王五在联合国表扬了。<br>Bad: 胡大爷被在联合国表扬了。 |
| multiple edits: bad deletes 张婶; bad inserts 张婶 | BEI_construction_b | 6 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 张婶被陈大姐的下属给批评了。<br>Bad: 被陈大姐的下属张婶给批评了。 |
| 肚 → 杯 | passive_body_part | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 我们被宋女士检查了肚子。<br>Bad: 我们被宋女士检查了杯子。 |
| 鼻子 → 衣服 | passive_body_part | 6 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 刘先生被她检查了鼻子。<br>Bad: 刘先生被她检查了衣服。 |
| multiple edits: bad deletes 火车上被; bad inserts 火车上 | BEI_deletion | 37 | 0.7568 | 0.9189 | -0.1622 | 0.0000 | Good: 火车上被陈大姐放满了电冰箱。<br>Bad: 陈大姐放满了火车上电冰箱。 |
| 鼻子 → 手套 | passive_body_part | 20 | 1.0000 | 0.8500 | +0.1500 | 0.0000 | Good: 他被何太太包扎了鼻子。<br>Bad: 他被何太太包扎了手套。 |
| bad deletes 你 | passive_agent_deletion_long_right_b | 15 | 0.7333 | 0.8667 | -0.1333 | 0.0000 | Good: 小王被你派何太太嘉奖了。<br>Bad: 小王被派何太太嘉奖了。 |
| multiple edits: bad deletes 他; bad inserts 他 | BEI_preposition | 31 | 0.6774 | 0.5484 | +0.1290 | 0.0000 | Good: 他被张婶批评了。<br>Bad: 被张婶他批评了。 |
| multiple edits: bad deletes 冯大哥; bad inserts 冯大哥 | BEI_construction_b | 8 | 0.6250 | 0.5000 | +0.1250 | 0.0000 | Good: 冯大哥被你们的儿子给批评了。<br>Bad: 被你们的儿子冯大哥给批评了。 |
| multiple edits: bad inserts 被吴太太; bad deletes 被吴太太 | BEI_construction_a | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 另外十头大象被吴太太给捕捉了。<br>Bad: 被吴太太另外十头大象给捕捉了。 |
| 耳朵 → 袜子 | passive_body_part | 8 | 0.0000 | 0.1250 | -0.1250 | 0.0000 | Good: 小王被他们检查了耳朵。<br>Bad: 小王被他们检查了袜子。 |
| 肚 → 裤 | passive_body_part | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 胡大爷被张夫人检查了肚子。<br>Bad: 胡大爷被张夫人检查了裤子。 |
| bad deletes 王姨 | passive_agent_deletion_long_right_b | 9 | 0.8889 | 1.0000 | -0.1111 | 0.0000 | Good: 张夫人被王姨派李四呵斥了。<br>Bad: 张夫人被派李四呵斥了。 |
| multiple edits: bad inserts 被赵大爷; bad deletes 被赵大爷 | BEI_construction_a | 9 | 1.0000 | 0.8889 | +0.1111 | 0.0000 | Good: 何太太的这五个玻璃珠被赵大爷给弹了。<br>Bad: 被赵大爷何太太的这五个玻璃珠给弹了。 |
| bad deletes 她 | passive_agent_deletion_short | 36 | 0.1667 | 0.0556 | +0.1111 | 0.0000 | Good: 徐小姐被她在欧盟批评了。<br>Bad: 徐小姐被在欧盟批评了。 |
| bad deletes 我 | passive_agent_deletion_long_right_a | 300 | 0.9200 | 0.8100 | +0.1100 | 0.0000 | Good: 那些花卷被我叫冯大哥请赵大爷托胡大爷给了。<br>Bad: 那些花卷被叫冯大哥请赵大爷托胡大爷给了。 |
| multiple edits: bad deletes 他们; bad inserts 他们 | BEI_construction_b | 19 | 0.7895 | 0.8947 | -0.1053 | 0.0000 | Good: 他们被这两位上级给安慰了。<br>Bad: 被这两位上级他们给安慰了。 |
| multiple edits: bad inserts 被她们; bad deletes 被她们 | BEI_construction_a | 23 | 1.0000 | 0.9130 | +0.0870 | 0.0000 | Good: 这个开瓶器被她们给买了。<br>Bad: 被她们这个开瓶器给买了。 |
| multiple edits: bad deletes 我们; bad inserts 我们 | BEI_preposition | 36 | 1.0000 | 0.9167 | +0.0833 | 0.0000 | Good: 我们被张先生呵斥了。<br>Bad: 被张先生我们呵斥了。 |
| multiple edits: bad deletes 卡车上被; bad inserts 卡车上 | BEI_deletion | 32 | 0.6562 | 0.5938 | +0.0625 | 0.0000 | Good: 卡车上被张先生扔满了作业。<br>Bad: 张先生扔满了卡车上作业。 |
| multiple edits: bad inserts 被我们; bad deletes 被我们 | BEI_construction_a | 19 | 0.9474 | 1.0000 | -0.0526 | 0.0000 | Good: 那杯咖啡被我们给喝了。<br>Bad: 被我们那杯咖啡给喝了。 |
| multiple edits: bad deletes 她们; bad inserts 她们 | BEI_construction_b | 20 | 1.0000 | 0.9500 | +0.0500 | 0.0000 | Good: 她们被那四位下属给提醒了。<br>Bad: 被那四位下属她们给提醒了。 |
| multiple edits: bad deletes 我; bad inserts 我 | BEI_preposition | 42 | 0.9762 | 0.9286 | +0.0476 | 0.0000 | Good: 我被赵大爷表扬了。<br>Bad: 被赵大爷我表扬了。 |
| multiple edits: bad deletes 你; bad inserts 你 | BEI_preposition | 43 | 0.9767 | 0.9302 | +0.0465 | 0.0000 | Good: 你被胡大爷表扬了。<br>Bad: 被胡大爷你表扬了。 |
| multiple edits: bad inserts 被我; bad deletes 被我 | BEI_construction_a | 22 | 0.9091 | 0.9545 | -0.0455 | 0.0000 | Good: 另外三头牛被我给屠宰了。<br>Bad: 被我另外三头牛给屠宰了。 |
| bad deletes 他们 | passive_agent_deletion_short | 22 | 0.0455 | 0.0909 | -0.0455 | 0.0000 | Good: 赵大爷被他们在欧盟约束了。<br>Bad: 赵大爷被在欧盟约束了。 |
| bad deletes 我们 | passive_agent_deletion_short | 24 | 0.7083 | 0.6667 | +0.0417 | 0.0000 | Good: 王先生被我们在国际足联夸奖了。<br>Bad: 王先生被在国际足联夸奖了。 |
| bad deletes 他们 | passive_suo | 24 | 1.0000 | 0.9583 | +0.0417 | 0.0000 | Good: 这个东西不可以被他们所知道。<br>Bad: 这个东西不可以被所知道。 |
| bad deletes 我 | passive_agent_deletion_long_right_b | 25 | 1.0000 | 0.9600 | +0.0400 | 0.0000 | Good: 张三被我派杨大哥提醒了。<br>Bad: 张三被派杨大哥提醒了。 |
| multiple edits: bad deletes 她们; bad inserts 她们 | BEI_preposition | 29 | 0.9655 | 0.9310 | +0.0345 | 0.0000 | Good: 她们被冯大哥伤害了。<br>Bad: 被冯大哥她们伤害了。 |
| multiple edits: bad deletes 你们; bad inserts 你们 | BEI_preposition | 50 | 0.9600 | 0.9400 | +0.0200 | 0.0000 | Good: 你们被小王表扬了。<br>Bad: 被小王你们表扬了。 |
| bad deletes 外国人 | passive_agent_deletion_long_left | 54 | 0.9815 | 0.9630 | +0.0185 | 0.0000 | Good: 刘先生今天对你的行为难免会被外国人所批评。<br>Bad: 刘先生今天对你的行为难免会被所批评。 |
| bad deletes 他人 | passive_agent_deletion_long_left | 84 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨今年对她的行为很可能会被他人所反感。<br>Bad: 王姨今年对她的行为很可能会被所反感。 |
| bad deletes 别人 | passive_agent_deletion_long_left | 61 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太刚刚对她们的理念容易被别人所支持。<br>Bad: 吴太太刚刚对她们的理念容易被所支持。 |
| bad deletes 外人 | passive_agent_deletion_long_left | 55 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生今年对我们的行为容易被外人所嫌弃。<br>Bad: 刘先生今年对我们的行为容易被所嫌弃。 |
| bad deletes 其他人 | passive_agent_deletion_long_left | 46 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥今天对你们的行为难免会被其他人所厌恶。<br>Bad: 杨大哥今天对你们的行为难免会被所厌恶。 |
| bad deletes 你们 | passive_agent_deletion_long_right_b | 24 | 0.4583 | 0.4583 | +0.0000 | 0.0000 | Good: 陈大姐被你们派张婶称赞了。<br>Bad: 陈大姐被派张婶称赞了。 |
| bad deletes 我 | passive_suo | 20 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个新闻不应该被我所了解。<br>Bad: 这个新闻不应该被所了解。 |
| multiple edits: bad deletes 你们; bad inserts 你们 | BEI_construction_b | 20 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们被那十位老板给批评了。<br>Bad: 被那十位老板你们给批评了。 |
| bad deletes 我们 | passive_suo | 19 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个事不应该被我们所了解。<br>Bad: 那个事不应该被所了解。 |
| multiple edits: bad deletes 你; bad inserts 你 | BEI_construction_b | 19 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你被那六位顾客给欺负了。<br>Bad: 被那六位顾客你给欺负了。 |
| bad deletes 你们 | passive_suo | 18 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个信息不应该被你们所知晓。<br>Bad: 那个信息不应该被所知晓。 |
| multiple edits: bad deletes 我; bad inserts 我 | BEI_construction_b | 17 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我被那九位员工给批评了。<br>Bad: 被那九位员工我给批评了。 |
| bad deletes 他 | passive_agent_deletion_short | 16 | 0.3125 | 0.3125 | +0.0000 | 0.0000 | Good: 王五被他在国际足联控制了。<br>Bad: 王五被在国际足联控制了。 |
| bad deletes 你 | passive_suo | 15 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个新闻不可以被你所知道。<br>Bad: 那个新闻不可以被所知道。 |
| multiple edits: bad deletes 她; bad inserts 她 | BEI_construction_b | 14 | 0.9286 | 0.9286 | +0.0000 | 0.0000 | Good: 她被那个演奏员给打劫了。<br>Bad: 被那个演奏员她给打劫了。 |
| multiple edits: bad deletes 我们; bad inserts 我们 | BEI_construction_b | 14 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们被这三位学生给照顾了。<br>Bad: 被这三位学生我们给照顾了。 |
| bad deletes 冯大哥 | passive_suo | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个秘密不可以被冯大哥所知道。<br>Bad: 这个秘密不可以被所知道。 |
| bad deletes 张三 | passive_agent_deletion_long_right_b | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生被张三派王大娘教育了。<br>Bad: 李先生被派王大娘教育了。 |
| multiple edits: bad deletes 张先生; bad inserts 张先生 | BEI_construction_b | 9 | 0.5556 | 0.5556 | +0.0000 | 0.0000 | Good: 张先生被另外十个奴隶给批评了。<br>Bad: 被另外十个奴隶张先生给批评了。 |
| 肚子 → 教材 | passive_body_part | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我被何太太包扎了肚子。<br>Bad: 我被何太太包扎了教材。 |
| 鼻子 → 教材 | passive_body_part | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷被她们检查了鼻子。<br>Bad: 赵大爷被她们检查了教材。 |
| bad deletes 王大娘 | passive_agent_deletion_long_right_b | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐被王大娘派周大妈欺负了。<br>Bad: 陈大姐被派周大妈欺负了。 |
| bad deletes 赵大爷 | passive_agent_deletion_long_right_b | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生被赵大爷派王小姐伤害了。<br>Bad: 王先生被派王小姐伤害了。 |
| bad deletes 陈大姐 | passive_agent_deletion_long_right_b | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太被陈大姐派张夫人约束了。<br>Bad: 吴太太被派张夫人约束了。 |
| 肚 → 裙 | passive_body_part | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王被她们包扎了肚子。<br>Bad: 小王被她们包扎了裙子。 |
| bad deletes 杨大哥 | passive_agent_deletion_long_right_b | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太被杨大哥派小明诽谤了。<br>Bad: 李太太被派小明诽谤了。 |
| multiple edits: bad deletes 吴太太; bad inserts 吴太太 | BEI_construction_b | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太被张婶的上级给伤害了。<br>Bad: 被张婶的上级吴太太给伤害了。 |
| multiple edits: bad deletes 宋女士; bad inserts 宋女士 | BEI_construction_b | 7 | 0.8571 | 0.8571 | +0.0000 | 0.0000 | Good: 宋女士被那个领导给表扬了。<br>Bad: 被那个领导宋女士给表扬了。 |
| multiple edits: bad deletes 徐小姐; bad inserts 徐小姐 | BEI_construction_b | 7 | 0.8571 | 0.8571 | +0.0000 | 0.0000 | Good: 徐小姐被这位服务员给欺负了。<br>Bad: 被这位服务员徐小姐给欺负了。 |
| 鼻子 → 作业 | passive_body_part | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生被他包扎了鼻子。<br>Bad: 刘先生被他包扎了作业。 |
| bad deletes 冯大哥 | passive_agent_deletion_short | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生被冯大哥在世界银行表扬了。<br>Bad: 王先生被在世界银行表扬了。 |
| bad deletes 王小姐 | passive_agent_deletion_long_right_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四被王小姐派张婶称赞了。<br>Bad: 李四被派张婶称赞了。 |
| bad deletes 胡大爷 | passive_agent_deletion_long_right_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生被胡大爷派宋女士称赞了。<br>Bad: 李先生被派宋女士称赞了。 |
| multiple edits: bad inserts 被刘先生; bad deletes 被刘先生 | BEI_construction_a | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的那张桌子被刘先生给搬了。<br>Bad: 被刘先生我们的那张桌子给搬了。 |
| multiple edits: bad inserts 被小王; bad deletes 被小王 | BEI_construction_a | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 那块糖果被小王给吃了。<br>Bad: 被小王那块糖果给吃了。 |
| multiple edits: bad inserts 被杨大哥; bad deletes 被杨大哥 | BEI_construction_a | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 这块巧克力被杨大哥给吃了。<br>Bad: 被杨大哥这块巧克力给吃了。 |
| bad deletes 小王 | passive_agent_deletion_short | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷被小王在非洲联盟伤害了。<br>Bad: 胡大爷被在非洲联盟伤害了。 |
| bad deletes 张先生 | passive_agent_deletion_long_right_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷被张先生派王先生责备了。<br>Bad: 胡大爷被派王先生责备了。 |
| bad deletes 张先生 | passive_agent_deletion_short | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘被张先生在非洲联盟责备了。<br>Bad: 王大娘被在非洲联盟责备了。 |
| bad deletes 杨大哥 | passive_agent_deletion_short | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人被杨大哥在北约教育了。<br>Bad: 张夫人被在北约教育了。 |
| bad deletes 杨大哥 | passive_suo | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个事情不可以被杨大哥所知晓。<br>Bad: 这个事情不可以被所知晓。 |
| bad deletes 王大娘 | passive_agent_deletion_short | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三被王大娘在欧盟批评了。<br>Bad: 张三被在欧盟批评了。 |
| bad deletes 胡大爷 | passive_agent_deletion_short | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五被胡大爷在欧盟打劫了。<br>Bad: 王五被在欧盟打劫了。 |
| bad deletes 郑大妈 | passive_agent_deletion_long_right_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥被郑大妈派张先生安慰了。<br>Bad: 冯大哥被派张先生安慰了。 |
| multiple edits: bad inserts 被周大妈; bad deletes 被周大妈 | BEI_construction_a | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的这只鸭被周大妈给炖了。<br>Bad: 被周大妈她们的这只鸭给炖了。 |
| multiple edits: bad inserts 被郑大妈; bad deletes 被郑大妈 | BEI_construction_a | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘的那六本书被郑大妈给写了。<br>Bad: 被郑大妈王大娘的那六本书给写了。 |
| 心脏 → 杯子 | passive_body_part | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐被小王检查了心脏。<br>Bad: 陈大姐被小王检查了杯子。 |
| bad deletes 冯大哥 | passive_agent_deletion_long_right_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生被冯大哥派王大娘称赞了。<br>Bad: 李先生被派王大娘称赞了。 |
| bad deletes 小王 | passive_agent_deletion_long_right_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐被小王派李太太原谅了。<br>Bad: 王小姐被派李太太原谅了。 |
| bad deletes 徐小姐 | passive_agent_deletion_long_right_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐被徐小姐派胡大爷照顾了。<br>Bad: 陈大姐被派胡大爷照顾了。 |
| bad deletes 李太太 | passive_suo | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个信息不应该被李太太所了解。<br>Bad: 这个信息不应该被所了解。 |
| bad deletes 王先生 | passive_agent_deletion_long_right_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太被王先生派张三欺骗了。<br>Bad: 吴太太被派张三欺骗了。 |
| bad deletes 王先生 | passive_agent_deletion_short | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥被王先生在议会控制了。<br>Bad: 杨大哥被在议会控制了。 |
| multiple edits: bad deletes 杨大哥; bad inserts 杨大哥 | BEI_construction_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥被那位学生给欺负了。<br>Bad: 被那位学生杨大哥给欺负了。 |
| multiple edits: bad deletes 王五; bad inserts 王五 | BEI_construction_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五被另外九个姐姐给批评了。<br>Bad: 被另外九个姐姐王五给批评了。 |
| multiple edits: bad inserts 被张夫人; bad deletes 被张夫人 | BEI_construction_a | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那块糖果被张夫人给吃了。<br>Bad: 被张夫人那块糖果给吃了。 |
| 心脏 → 裤子 | passive_body_part | 4 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们被张婶检查了心脏。<br>Bad: 他们被张婶检查了裤子。 |
| 控制 → 呼吸 | passive_intransitive | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶遭到他们的下属控制。<br>Bad: 张婶遭到他们的下属呼吸。 |
| 耳朵 → 教材 | passive_body_part | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明被她检查了耳朵。<br>Bad: 小明被她检查了教材。 |
| 耳朵 → 衣服 | passive_body_part | 4 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 宋女士被她检查了耳朵。<br>Bad: 宋女士被她检查了衣服。 |
| bad deletes 何太太 | passive_agent_deletion_short | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐被何太太在奥委会夸奖了。<br>Bad: 徐小姐被在奥委会夸奖了。 |
| bad deletes 李太太 | passive_agent_deletion_short | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生被李太太在非洲联盟称赞了。<br>Bad: 张先生被在非洲联盟称赞了。 |
| bad deletes 郑大妈 | passive_agent_deletion_short | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太被郑大妈在议会原谅了。<br>Bad: 何太太被在议会原谅了。 |
| bad deletes 陈大姐 | passive_agent_deletion_short | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明被陈大姐在议会约束了。<br>Bad: 小明被在议会约束了。 |
| multiple edits: bad deletes 小明; bad inserts 小明 | BEI_construction_b | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明被那九位上级给欺骗了。<br>Bad: 被那九位上级小明给欺骗了。 |
| multiple edits: bad deletes 小王; bad inserts 小王 | BEI_construction_b | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王被徐小姐的老师给欺负了。<br>Bad: 被徐小姐的老师小王给欺负了。 |
| multiple edits: bad inserts 被宋女士; bad deletes 被宋女士 | BEI_construction_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她的这六杯咖啡被宋女士给喝了。<br>Bad: 被宋女士她的这六杯咖啡给喝了。 |
| multiple edits: bad inserts 被小明; bad deletes 被小明 | BEI_construction_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外十桶啤酒被小明给买了。<br>Bad: 被小明另外十桶啤酒给买了。 |
| multiple edits: bad inserts 被胡大爷; bad deletes 被胡大爷 | BEI_construction_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九块糖果被胡大爷给吃了。<br>Bad: 被胡大爷那九块糖果给吃了。 |
| 心脏 → 作业 | passive_body_part | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你被王大娘检查了心脏。<br>Bad: 你被王大娘检查了作业。 |
| 心脏 → 教材 | passive_body_part | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生被张三检查了心脏。<br>Bad: 刘先生被张三检查了教材。 |
| 批评 → 普通 | passive_no_adj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生的母亲受到刘先生批评了。<br>Bad: 张先生的母亲受到刘先生普通了。 |
| 欺负 → 高昂 | passive_no_adj | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个朋友被李先生欺负了。<br>Bad: 这个朋友被李先生高昂了。 |
| bad deletes 吴太太 | passive_agent_deletion_short | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生被吴太太在奥委会教育了。<br>Bad: 刘先生被在奥委会教育了。 |
| bad deletes 周大妈 | passive_agent_deletion_long_right_b | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥被周大妈派徐小姐控制了。<br>Bad: 杨大哥被派徐小姐控制了。 |
| bad deletes 周大妈 | passive_agent_deletion_short | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四被周大妈在欧盟奖励了。<br>Bad: 李四被在欧盟奖励了。 |
| bad deletes 张三 | passive_agent_deletion_short | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生被张三在亚投行打劫了。<br>Bad: 王先生被在亚投行打劫了。 |
| bad deletes 王小姐 | passive_agent_deletion_short | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明被王小姐在亚投行教育了。<br>Bad: 小明被在亚投行教育了。 |
| multiple edits: bad inserts 被张先生; bad deletes 被张先生 | BEI_construction_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一只老虎被张先生给麻醉了。<br>Bad: 被张先生另外一只老虎给麻醉了。 |
| multiple edits: bad inserts 被王小姐; bad deletes 被王小姐 | BEI_construction_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥的那瓶矿泉水被王小姐给喝了。<br>Bad: 被王小姐冯大哥的那瓶矿泉水给喝了。 |
| 喜欢 → 溜走 | passive_intransitive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生被那十位员工喜欢。<br>Bad: 李先生被那十位员工溜走。 |
| 埋怨 → 爬行 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈被他们的爸爸埋怨。<br>Bad: 周大妈被他们的爸爸爬行。 |
| 夸奖 → 难过 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈的上级被她夸奖了。<br>Bad: 郑大妈的上级被她难过了。 |
| 嫌弃 → 唱歌 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐受到这九个司机嫌弃。<br>Bad: 王小姐受到这九个司机唱歌。 |
| 安慰 → 普通 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位演员受到小王安慰了。<br>Bad: 那位演员受到小王普通了。 |
| 心脏 → 被子 | passive_body_part | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们被她检查了心脏。<br>Bad: 你们被她检查了被子。 |
| 打劫 → 鲜嫩 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨的兄弟遭到王先生打劫了。<br>Bad: 王姨的兄弟遭到王先生鲜嫩了。 |
| 批评 → 听课 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太遭到我的学生批评。<br>Bad: 李太太遭到我的学生听课。 |
| 批评 → 年轻 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三的弟弟被你批评了。<br>Bad: 张三的弟弟被你年轻了。 |
| 批评 → 忧郁 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生的老板被我们批评了。<br>Bad: 王先生的老板被我们忧郁了。 |
| 批评 → 苦恼 | passive_no_adj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那六位演员遭到你批评了。<br>Bad: 那六位演员遭到你苦恼了。 |
| 拥护 → 走路 | passive_intransitive | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐受到她们的老板拥护。<br>Bad: 陈大姐受到她们的老板走路。 |
| 排挤 → 看戏 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘被我们的老师排挤。<br>Bad: 王大娘被我们的老师看戏。 |
| 控制 → 昂贵 | passive_no_adj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位员工遭到冯大哥控制了。<br>Bad: 这位员工遭到冯大哥昂贵了。 |
| 控制 → 热烈 | passive_no_adj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位音乐家受到王姨控制了。<br>Bad: 那位音乐家受到王姨热烈了。 |
| 控制 → 甘甜 | passive_no_adj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那个儿子受到你控制了。<br>Bad: 那个儿子受到你甘甜了。 |
| 支持 → 运动 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人受到那个记者支持。<br>Bad: 张夫人受到那个记者运动。 |
| 欺负 → 便宜 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三的老师受到他们欺负了。<br>Bad: 张三的老师受到他们便宜了。 |
| 欺负 → 舒缓 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个罪犯被张婶欺负了。<br>Bad: 那个罪犯被张婶舒缓了。 |
| 照顾 → 无聊 | passive_no_adj | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位下属被他照顾了。<br>Bad: 那位下属被他无聊了。 |
| 照顾 → 精致 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生的母亲被小明照顾了。<br>Bad: 李先生的母亲被小明精致了。 |
| 爱戴 → 呼吸 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈受到他的老师爱戴。<br>Bad: 周大妈受到他的老师呼吸。 |
| 爱戴 → 看戏 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈受到这七位舞者爱戴。<br>Bad: 郑大妈受到这七位舞者看戏。 |
| 眼睛 → 作业 | passive_body_part | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥被她检查了眼睛。<br>Bad: 杨大哥被她检查了作业。 |
| 眼睛 → 教材 | passive_body_part | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶被我检查了眼睛。<br>Bad: 张婶被我检查了教材。 |
| 眼睛 → 袜子 | passive_body_part | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生被何太太检查了眼睛。<br>Bad: 刘先生被何太太检查了袜子。 |
| 称赞 → 特殊 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一个舞者被王五称赞了。<br>Bad: 另外一个舞者被王五特殊了。 |
| 耳朵 → 桌子 | passive_body_part | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘被李太太包扎了耳朵。<br>Bad: 王大娘被李太太包扎了桌子。 |
| 耳朵 → 被子 | passive_body_part | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生被你们检查了耳朵。<br>Bad: 刘先生被你们检查了被子。 |
| 肚子 → 作业 | passive_body_part | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生被小明包扎了肚子。<br>Bad: 张先生被小明包扎了作业。 |
| 表扬 → 启程 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐受到这位上级表扬。<br>Bad: 徐小姐受到这位上级启程。 |
| 表扬 → 方便 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位服务员被张婶表扬了。<br>Bad: 这位服务员被张婶方便了。 |
| 表扬 → 昂贵 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外九个舞者被我表扬了。<br>Bad: 另外九个舞者被我昂贵了。 |
| 表扬 → 特殊 | passive_no_adj | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个舞者受到何太太表扬了。<br>Bad: 那个舞者受到何太太特殊了。 |
| 表扬 → 过去 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士受到那四位舞者表扬。<br>Bad: 宋女士受到那四位舞者过去。 |
| 赞成 → 过去 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生受到那位员工赞成。<br>Bad: 刘先生受到那位员工过去。 |
| 辩护 → 呼吸 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈受到她们的父亲辩护。<br>Bad: 郑大妈受到她们的父亲呼吸。 |
| 青睐 → 起飞 | passive_intransitive | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶被他的同事青睐。<br>Bad: 张婶被他的同事起飞。 |
| 鼓励 → 低沉 | passive_no_adj | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 这位顾客受到刘先生鼓励了。<br>Bad: 这位顾客受到刘先生低沉了。 |
| bad deletes 张婶 | passive_agent_deletion_short | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三被张婶在国际足联教育了。<br>Bad: 张三被在国际足联教育了。 |
| multiple edits: bad deletes 李先生; bad inserts 生李先 | BEI_construction_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生被这位学生给批评了。<br>Bad: 被这位学生李先生给批评了。 |
| multiple edits: bad deletes 李四; bad inserts 李四 | BEI_construction_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四被我的妹妹给安慰了。<br>Bad: 被我的妹妹李四给安慰了。 |
| 伤害 → 欢快 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥的老板被他们伤害了。<br>Bad: 冯大哥的老板被他们欢快了。 |
| 伤害 → 浓郁 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外五位打工人遭到我们伤害了。<br>Bad: 另外五位打工人遭到我们浓郁了。 |
| 伤害 → 深刻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位工人受到我伤害了。<br>Bad: 那位工人受到我深刻了。 |
| 伤害 → 狂野 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的爸爸被王小姐伤害了。<br>Bad: 他的爸爸被王小姐狂野了。 |
| 伤害 → 看戏 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈受到这位老师伤害。<br>Bad: 郑大妈受到这位老师看戏。 |
| 伤害 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷的上级被徐小姐伤害了。<br>Bad: 赵大爷的上级被徐小姐酸爽了。 |
| 伤害 → 鲜嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位工人被她们伤害了。<br>Bad: 那位工人被她们鲜嫩了。 |
| 厌恶 → 停下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士遭到她们的姐姐厌恶。<br>Bad: 宋女士遭到她们的姐姐停下。 |
| 厌恶 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷遭到那个上级厌恶。<br>Bad: 赵大爷遭到那个上级启程。 |
| 厌恶 → 坐下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶受到那十位上级厌恶。<br>Bad: 张婶受到那十位上级坐下。 |
| 厌恶 → 打架 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太被那个打工人厌恶。<br>Bad: 李太太被那个打工人打架。 |
| 厌恶 → 溜走 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太受到你们的领导厌恶。<br>Bad: 吴太太受到你们的领导溜走。 |
| 厌恶 → 过来 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四遭到张婶的上级厌恶。<br>Bad: 李四遭到张婶的上级过来。 |
| 厌恶 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五被那个音乐家厌恶。<br>Bad: 王五被那个音乐家运动。 |
| 厌恶 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五被这一个消费者厌恶。<br>Bad: 王五被这一个消费者闲逛。 |
| 厌恶 → 颤抖 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘遭到李太太的下属厌恶。<br>Bad: 王大娘遭到李太太的下属颤抖。 |
| 原谅 → 凶猛 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位钢琴家受到张夫人原谅了。<br>Bad: 这位钢琴家受到张夫人凶猛了。 |
| 原谅 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥受到这五位舞者原谅。<br>Bad: 冯大哥受到这五位舞者唱歌。 |
| 原谅 → 忧郁 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这两位司机被小明原谅了。<br>Bad: 这两位司机被小明忧郁了。 |
| 原谅 → 成熟 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七个罪犯受到周大妈原谅了。<br>Bad: 这七个罪犯受到周大妈成熟了。 |
| 原谅 → 打架 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐受到那位老板原谅。<br>Bad: 陈大姐受到那位老板打架。 |
| 原谅 → 浅显 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位领导受到我们原谅了。<br>Bad: 那位领导受到我们浅显了。 |
| 原谅 → 浓郁 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位舞者受到他原谅了。<br>Bad: 那位舞者受到他浓郁了。 |
| 原谅 → 热烈 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的上级受到你们原谅了。<br>Bad: 我们的上级受到你们热烈了。 |
| 原谅 → 粗旷 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王的领导被他们原谅了。<br>Bad: 小王的领导被他们粗旷了。 |
| 原谅 → 舒缓 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三位同事受到她们原谅了。<br>Bad: 这三位同事受到她们舒缓了。 |
| 原谅 → 辛辣 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位同事受到你们原谅了。<br>Bad: 那位同事受到你们辛辣了。 |
| 原谅 → 酥脆 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位员工受到你们原谅了。<br>Bad: 那位员工受到你们酥脆了。 |
| 原谅 → 酥软 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的儿子被他们原谅了。<br>Bad: 他的儿子被他们酥软了。 |
| 原谅 → 醇厚 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的员工受到他原谅了。<br>Bad: 你们的员工受到他醇厚了。 |
| 原谅 → 鲜嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个顾客受到他们原谅了。<br>Bad: 那个顾客受到他们鲜嫩了。 |
| 反感 → 停下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明遭到这个奴隶反感。<br>Bad: 小明遭到这个奴隶停下。 |
| 反感 → 健身 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘被这三个奴隶反感。<br>Bad: 王大娘被这三个奴隶健身。 |
| 反感 → 出发 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈受到这四位上级反感。<br>Bad: 周大妈受到这四位上级出发。 |
| 反感 → 听课 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李四受到那位音乐家反感。<br>Bad: 李四受到那位音乐家听课。 |
| 反感 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷遭到那位上级反感。<br>Bad: 赵大爷遭到那位上级唱歌。 |
| 反感 → 爬行 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐被另外三个罪犯反感。<br>Bad: 王小姐被另外三个罪犯爬行。 |
| 反感 → 跳舞 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生被李先生的兄弟反感。<br>Bad: 张先生被李先生的兄弟跳舞。 |
| 反感 → 躺下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐被那六个小孩反感。<br>Bad: 陈大姐被那六个小孩躺下。 |
| 反感 → 过去 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐被另外两位父亲反感。<br>Bad: 陈大姐被另外两位父亲过去。 |
| 反感 → 过来 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐被陈大姐的下属反感。<br>Bad: 陈大姐被陈大姐的下属过来。 |
| 反驳 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈遭到这九个舞者反驳。<br>Bad: 周大妈遭到这九个舞者启程。 |
| 呵斥 → 伤心 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个吉他手被我们呵斥了。<br>Bad: 这个吉他手被我们伤心了。 |
| 呵斥 → 入睡 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王被那位空姐呵斥。<br>Bad: 小王被那位空姐入睡。 |
| 呵斥 → 动听 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨的上级遭到宋女士呵斥了。<br>Bad: 王姨的上级遭到宋女士动听了。 |
| 呵斥 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈被她们的姐妹呵斥。<br>Bad: 周大妈被她们的姐妹唱歌。 |
| 呵斥 → 微笑 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷遭到王先生的领导呵斥。<br>Bad: 赵大爷遭到王先生的领导微笑。 |
| 呵斥 → 过来 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太被这两个工人呵斥。<br>Bad: 何太太被这两个工人过来。 |
| 呵斥 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨遭到王先生的老师呵斥。<br>Bad: 王姨遭到王先生的老师运动。 |
| 呵斥 → 鲜嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位母亲受到冯大哥呵斥了。<br>Bad: 那位母亲受到冯大哥鲜嫩了。 |
| 喜欢 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥受到他的老板喜欢。<br>Bad: 冯大哥受到他的老板启程。 |
| 喜欢 → 呼吸 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太被他的妈妈喜欢。<br>Bad: 吴太太被他的妈妈呼吸。 |
| 喜欢 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷被这九个上级喜欢。<br>Bad: 胡大爷被这九个上级唱歌。 |
| 喜欢 → 玩耍 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨被这个演奏员喜欢。<br>Bad: 王姨被这个演奏员玩耍。 |
| 喜欢 → 站立 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶受到那四位空姐喜欢。<br>Bad: 张婶受到那四位空姐站立。 |
| 喜欢 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈被这个消费者喜欢。<br>Bad: 周大妈被这个消费者起飞。 |
| 喜欢 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四受到这六个朋友喜欢。<br>Bad: 李四受到这六个朋友运动。 |
| 嘉奖 → 停下 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生受到这位上级嘉奖。<br>Bad: 张先生受到这位上级停下。 |
| 嘉奖 → 冷静 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个弟弟受到你们嘉奖了。<br>Bad: 这个弟弟受到你们冷静了。 |
| 嘉奖 → 微笑 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘被这个姐姐嘉奖。<br>Bad: 王大娘被这个姐姐微笑。 |
| 嘉奖 → 普通 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个服务员受到小王嘉奖了。<br>Bad: 这个服务员受到小王普通了。 |
| 嘉奖 → 深刻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐的老板被王姨嘉奖了。<br>Bad: 王小姐的老板被王姨深刻了。 |
| 嘉奖 → 滑嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的朋友受到你们嘉奖了。<br>Bad: 你的朋友受到你们滑嫩了。 |
| 嘉奖 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生被徐小姐的儿子嘉奖。<br>Bad: 刘先生被徐小姐的儿子起飞。 |
| 嘉奖 → 难过 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐的父亲受到你嘉奖了。<br>Bad: 王小姐的父亲受到你难过了。 |
| 埋怨 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘受到那八个记者埋怨。<br>Bad: 王大娘受到那八个记者偷听。 |
| 埋怨 → 出发 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥被冯大哥的父亲埋怨。<br>Bad: 杨大哥被冯大哥的父亲出发。 |
| 埋怨 → 叹息 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐受到吴太太的父亲埋怨。<br>Bad: 陈大姐受到吴太太的父亲叹息。 |
| 埋怨 → 坐下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生受到那位演员埋怨。<br>Bad: 张先生受到那位演员坐下。 |
| 埋怨 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五遭到我的老板埋怨。<br>Bad: 王五遭到我的老板游泳。 |
| 埋怨 → 溜走 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王被他们的领导埋怨。<br>Bad: 小王被他们的领导溜走。 |
| 埋怨 → 玩耍 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐受到他们的妈妈埋怨。<br>Bad: 徐小姐受到他们的妈妈玩耍。 |
| 埋怨 → 看戏 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四被吴太太的女儿埋怨。<br>Bad: 李四被吴太太的女儿看戏。 |
| 埋怨 → 躺下 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈被这六个演奏员埋怨。<br>Bad: 周大妈被这六个演奏员躺下。 |
| 埋怨 → 过去 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士被刘先生的下属埋怨。<br>Bad: 宋女士被刘先生的下属过去。 |
| 夸奖 → 优雅 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生的下属被你夸奖了。<br>Bad: 王先生的下属被你优雅了。 |
| 夸奖 → 伤心 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五的领导被徐小姐夸奖了。<br>Bad: 王五的领导被徐小姐伤心了。 |
| 夸奖 → 低沉 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们的上级受到他们夸奖了。<br>Bad: 我们的上级受到他们低沉了。 |
| 夸奖 → 年轻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他的领导被你们夸奖了。<br>Bad: 他的领导被你们年轻了。 |
| 夸奖 → 明确 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那七位钢琴家受到她们夸奖了。<br>Bad: 那七位钢琴家受到她们明确了。 |
| 夸奖 → 欢快 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六个记者被我夸奖了。<br>Bad: 这六个记者被我欢快了。 |
| 夸奖 → 特殊 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个妹妹被王大娘夸奖了。<br>Bad: 那个妹妹被王大娘特殊了。 |
| 夸奖 → 狂野 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位下属受到小王夸奖了。<br>Bad: 那位下属受到小王狂野了。 |
| 夸奖 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外三位空姐受到赵大爷夸奖了。<br>Bad: 另外三位空姐受到赵大爷酸爽了。 |
| 夸奖 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人被张夫人的下属夸奖。<br>Bad: 张夫人被张夫人的下属闲逛。 |
| 夸奖 → 高昂 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外九位父亲被张婶夸奖了。<br>Bad: 另外九位父亲被张婶高昂了。 |
| 奖励 → 入睡 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷受到那个工人奖励。<br>Bad: 赵大爷受到那个工人入睡。 |
| 奖励 → 叹息 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生被这位舞者奖励。<br>Bad: 张先生被这位舞者叹息。 |
| 奖励 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶受到另外一个记者奖励。<br>Bad: 张婶受到另外一个记者启程。 |
| 奖励 → 愤怒 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的同事被他奖励了。<br>Bad: 他们的同事被他愤怒了。 |
| 奖励 → 深刻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三个小孩受到吴太太奖励了。<br>Bad: 这三个小孩受到吴太太深刻了。 |
| 奖励 → 清淡 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈的儿子受到何太太奖励了。<br>Bad: 周大妈的儿子受到何太太清淡了。 |
| 奖励 → 看戏 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太被那六位母亲奖励。<br>Bad: 李太太被那六位母亲看戏。 |
| 奖励 → 舒缓 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那八位同事受到小王奖励了。<br>Bad: 那八位同事受到小王舒缓了。 |
| 奖励 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位打工人受到小王奖励了。<br>Bad: 那位打工人受到小王酸爽了。 |
| 嫌弃 → 健身 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生受到那个妹妹嫌弃。<br>Bad: 张先生受到那个妹妹健身。 |
| 嫌弃 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生受到郑大妈的母亲嫌弃。<br>Bad: 张先生受到郑大妈的母亲启程。 |
| 嫌弃 → 呼吸 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘遭到这五位钢琴家嫌弃。<br>Bad: 王大娘遭到这五位钢琴家呼吸。 |
| 嫌弃 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王受到李太太的员工嫌弃。<br>Bad: 小王受到李太太的员工游泳。 |
| 嫌弃 → 站立 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太被那个顾客嫌弃。<br>Bad: 吴太太被那个顾客站立。 |
| 嫌弃 → 走路 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人遭到那位演奏员嫌弃。<br>Bad: 张夫人遭到那位演奏员走路。 |
| 嫌弃 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘受到另外九位母亲嫌弃。<br>Bad: 王大娘受到另外九位母亲起飞。 |
| 嫌弃 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈被张婶的姐姐嫌弃。<br>Bad: 周大妈被张婶的姐姐运动。 |
| 嫌弃 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太被这个上级嫌弃。<br>Bad: 李太太被这个上级闲逛。 |
| 安慰 → 低沉 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位音乐家受到李太太安慰了。<br>Bad: 这位音乐家受到李太太低沉了。 |
| 安慰 → 保守 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个罪犯被胡大爷安慰了。<br>Bad: 那个罪犯被胡大爷保守了。 |
| 安慰 → 动听 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐的姐姐受到我安慰了。<br>Bad: 徐小姐的姐姐受到我动听了。 |
| 安慰 → 坐下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太受到那位老板安慰。<br>Bad: 吴太太受到那位老板坐下。 |
| 安慰 → 明确 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太的女儿受到张婶安慰了。<br>Bad: 何太太的女儿受到张婶明确了。 |
| 安慰 → 有趣 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个女儿受到郑大妈安慰了。<br>Bad: 那个女儿受到郑大妈有趣了。 |
| 安慰 → 深情 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那个司机受到李四安慰了。<br>Bad: 那个司机受到李四深情了。 |
| 安慰 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士被我们的妈妈安慰。<br>Bad: 宋女士被我们的妈妈游泳。 |
| 安慰 → 甘甜 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位老师被她们安慰了。<br>Bad: 那位老师被她们甘甜了。 |
| 安慰 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五被李四的同事安慰。<br>Bad: 王五被李四的同事运动。 |
| 安慰 → 酥软 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外八个服务员被我安慰了。<br>Bad: 另外八个服务员被我酥软了。 |
| 安慰 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那三位下属被周大妈安慰了。<br>Bad: 那三位下属被周大妈酸爽了。 |
| 宠爱 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘被赵大爷的爸爸宠爱。<br>Bad: 王大娘被赵大爷的爸爸启程。 |
| 宠爱 → 睡觉 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘受到这个妹妹宠爱。<br>Bad: 王大娘受到这个妹妹睡觉。 |
| 宠爱 → 站立 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈受到那个朋友宠爱。<br>Bad: 周大妈受到那个朋友站立。 |
| 宠爱 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太被那位吉他手宠爱。<br>Bad: 何太太被那位吉他手起飞。 |
| 宠爱 → 跑步 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四受到那九个小孩宠爱。<br>Bad: 李四受到那九个小孩跑步。 |
| 宠爱 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐受到这位打工人宠爱。<br>Bad: 王小姐受到这位打工人运动。 |
| 尊重 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生受到那个领导尊重。<br>Bad: 李先生受到那个领导起飞。 |
| 尊重 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明受到另外八位老板尊重。<br>Bad: 小明受到另外八位老板运动。 |
| 心脏 → 衣服 | passive_body_part | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你被她检查了心脏。<br>Bad: 你被她检查了衣服。 |
| 心脏 → 袜子 | passive_body_part | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈被他们检查了心脏。<br>Bad: 郑大妈被他们检查了袜子。 |
| 憎恨 → 健身 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨受到这四位空姐憎恨。<br>Bad: 王姨受到这四位空姐健身。 |
| 憎恨 → 偷听 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈遭到这个音乐家憎恨。<br>Bad: 郑大妈遭到这个音乐家偷听。 |
| 憎恨 → 出发 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明被那个罪犯憎恨。<br>Bad: 小明被那个罪犯出发。 |
| 憎恨 → 呼吸 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶遭到这两位舞者憎恨。<br>Bad: 张婶遭到这两位舞者呼吸。 |
| 憎恨 → 品茶 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生被那位老板憎恨。<br>Bad: 李先生被那位老板品茶。 |
| 憎恨 → 爬行 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐被这位下属憎恨。<br>Bad: 徐小姐被这位下属爬行。 |
| 憎恨 → 睡觉 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王受到这两个老板憎恨。<br>Bad: 小王受到这两个老板睡觉。 |
| 憎恨 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三遭到她们的兄弟憎恨。<br>Bad: 张三遭到她们的兄弟起飞。 |
| 憎恨 → 跑步 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生受到这九位记者憎恨。<br>Bad: 王先生受到这九位记者跑步。 |
| 憎恨 → 跳舞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生遭到徐小姐的老板憎恨。<br>Bad: 李先生遭到徐小姐的老板跳舞。 |
| 打劫 → 优雅 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那八个罪犯遭到他打劫了。<br>Bad: 那八个罪犯遭到他优雅了。 |
| 打劫 → 低沉 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外六个司机遭到宋女士打劫了。<br>Bad: 另外六个司机遭到宋女士低沉了。 |
| 打劫 → 呼吸 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太遭到那个消费者打劫。<br>Bad: 吴太太遭到那个消费者呼吸。 |
| 打劫 → 悠扬 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个舞者受到他打劫了。<br>Bad: 这个舞者受到他悠扬了。 |
| 打劫 → 方便 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的老板被王先生打劫了。<br>Bad: 她们的老板被王先生方便了。 |
| 打劫 → 昂贵 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一个领导受到小明打劫了。<br>Bad: 另外一个领导受到小明昂贵了。 |
| 打劫 → 普通 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外八位学生遭到徐小姐打劫了。<br>Bad: 另外八位学生遭到徐小姐普通了。 |
| 打劫 → 特殊 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位老师被王先生打劫了。<br>Bad: 那位老师被王先生特殊了。 |
| 打劫 → 狂放 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那十个小孩遭到胡大爷打劫了。<br>Bad: 那十个小孩遭到胡大爷狂放了。 |
| 打劫 → 精致 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六位记者遭到王姨打劫了。<br>Bad: 这六位记者遭到王姨精致了。 |
| 打劫 → 走路 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太遭到他们的哥哥打劫。<br>Bad: 吴太太遭到他们的哥哥走路。 |
| 打劫 → 酸甜 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王的老板被胡大爷打劫了。<br>Bad: 小王的老板被胡大爷酸甜了。 |
| 批判 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶受到周大妈的朋友批判。<br>Bad: 张婶受到周大妈的朋友偷听。 |
| 批判 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥受到你们的同事批判。<br>Bad: 冯大哥受到你们的同事游泳。 |
| 批评 → 优雅 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生的学生被张夫人批评了。<br>Bad: 张先生的学生被张夫人优雅了。 |
| 批评 → 停下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈遭到这个司机批评。<br>Bad: 郑大妈遭到这个司机停下。 |
| 批评 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张夫人被这两位下属批评。<br>Bad: 张夫人被这两位下属偷听。 |
| 批评 → 坎坷 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个舞者遭到徐小姐批评了。<br>Bad: 这个舞者遭到徐小姐坎坷了。 |
| 批评 → 坐下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥遭到宋女士的老板批评。<br>Bad: 冯大哥遭到宋女士的老板坐下。 |
| 批评 → 方便 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七位打工人被张夫人批评了。<br>Bad: 这七位打工人被张夫人方便了。 |
| 批评 → 无聊 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们的上级受到你们批评了。<br>Bad: 他们的上级受到你们无聊了。 |
| 批评 → 温驯 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外三位记者受到张婶批评了。<br>Bad: 另外三位记者受到张婶温驯了。 |
| 批评 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生受到那九个朋友批评。<br>Bad: 李先生受到那九个朋友游泳。 |
| 批评 → 滑嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那四个上级被我批评了。<br>Bad: 那四个上级被我滑嫩了。 |
| 批评 → 特殊 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你的父亲受到张三批评了。<br>Bad: 你的父亲受到张三特殊了。 |
| 批评 → 狂放 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位消费者被他们批评了。<br>Bad: 这位消费者被他们狂放了。 |
| 批评 → 玩耍 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太被我的爸爸批评。<br>Bad: 何太太被我的爸爸玩耍。 |
| 批评 → 过去 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太遭到这个舞者批评。<br>Bad: 何太太遭到这个舞者过去。 |
| 批评 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五受到她的父亲批评。<br>Bad: 王五受到她的父亲运动。 |
| 批评 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位工人遭到张夫人批评了。<br>Bad: 那位工人遭到张夫人酸爽了。 |
| 批评 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷被这五个领导批评。<br>Bad: 赵大爷被这五个领导闲逛。 |
| 批评 → 难过 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的领导遭到她们批评了。<br>Bad: 你们的领导遭到她们难过了。 |
| 批评 → 顺利 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个上级受到我们批评了。<br>Bad: 这个上级受到我们顺利了。 |
| 批评 → 颤抖 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐受到你们的弟弟批评。<br>Bad: 徐小姐受到你们的弟弟颤抖。 |
| 批评 → 鲜嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这两位上级被郑大妈批评了。<br>Bad: 这两位上级被郑大妈鲜嫩了。 |
| 批评 → 鲜美 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那三位打工人受到吴太太批评了。<br>Bad: 那三位打工人受到吴太太鲜美了。 |
| 抨击 → 跑步 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五遭到这个姐姐抨击。<br>Bad: 王五遭到这个姐姐跑步。 |
| 抨击 → 躺下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生被那七个打工人抨击。<br>Bad: 李先生被那七个打工人躺下。 |
| 拥护 → 叹息 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太受到这位老师拥护。<br>Bad: 何太太受到这位老师叹息。 |
| 拥护 → 听课 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太受到另外九个音乐家拥护。<br>Bad: 李太太受到另外九个音乐家听课。 |
| 拥护 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人被这九个记者拥护。<br>Bad: 张夫人被这九个记者起飞。 |
| 拥护 → 跑步 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈受到那个女儿拥护。<br>Bad: 郑大妈受到那个女儿跑步。 |
| 拥护 → 跳舞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五被这位下属拥护。<br>Bad: 王五被这位下属跳舞。 |
| 拥护 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐受到这位老师拥护。<br>Bad: 陈大姐受到这位老师运动。 |
| 拥护 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明受到她们的弟弟拥护。<br>Bad: 小明受到她们的弟弟闲逛。 |
| 拥护 → 颤抖 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生被这个女儿拥护。<br>Bad: 刘先生被这个女儿颤抖。 |
| 排挤 → 听课 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太遭到那位下属排挤。<br>Bad: 李太太遭到那位下属听课。 |
| 排挤 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太遭到另外五个服务员排挤。<br>Bad: 何太太遭到另外五个服务员唱歌。 |
| 排挤 → 微笑 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈遭到徐小姐的同事排挤。<br>Bad: 郑大妈遭到徐小姐的同事微笑。 |
| 排挤 → 爬行 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈被这三个演奏员排挤。<br>Bad: 周大妈被这三个演奏员爬行。 |
| 排挤 → 站立 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王五受到我们的老师排挤。<br>Bad: 王五受到我们的老师站立。 |
| 排挤 → 过来 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷遭到你的朋友排挤。<br>Bad: 赵大爷遭到你的朋友过来。 |
| 排挤 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨受到那个演奏员排挤。<br>Bad: 王姨受到那个演奏员运动。 |
| 控制 → 低沉 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位员工遭到刘先生控制了。<br>Bad: 那位员工遭到刘先生低沉了。 |
| 控制 → 便宜 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的姐妹受到我们控制了。<br>Bad: 李太太的姐妹受到我们便宜了。 |
| 控制 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈受到这个打工人控制。<br>Bad: 周大妈受到这个打工人偷听。 |
| 控制 → 失望 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位空姐受到李四控制了。<br>Bad: 那位空姐受到李四失望了。 |
| 控制 → 开心 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三的女儿被他控制了。<br>Bad: 张三的女儿被他开心了。 |
| 控制 → 特殊 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这个姐姐被赵大爷控制了。<br>Bad: 这个姐姐被赵大爷特殊了。 |
| 控制 → 狂放 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位下属遭到张夫人控制了。<br>Bad: 那位下属遭到张夫人狂放了。 |
| 控制 → 醇厚 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位打工人遭到你控制了。<br>Bad: 那位打工人遭到你醇厚了。 |
| 控制 → 难过 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 另外七个姐姐被小明控制了。<br>Bad: 另外七个姐姐被小明难过了。 |
| 推崇 → 微笑 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐受到那个朋友推崇。<br>Bad: 陈大姐受到那个朋友微笑。 |
| 推崇 → 溜走 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨受到王小姐的弟弟推崇。<br>Bad: 王姨受到王小姐的弟弟溜走。 |
| 推崇 → 看戏 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生被这九位服务员推崇。<br>Bad: 张先生被这九位服务员看戏。 |
| 推崇 → 睡觉 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐被这十个演奏员推崇。<br>Bad: 陈大姐被这十个演奏员睡觉。 |
| 推崇 → 跳舞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生被那个奴隶推崇。<br>Bad: 张先生被那个奴隶跳舞。 |
| 推崇 → 过去 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王被这四个舞者推崇。<br>Bad: 小王被这四个舞者过去。 |
| 推崇 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷受到另外八位上级推崇。<br>Bad: 赵大爷受到另外八位上级运动。 |
| 提醒 → 出发 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明受到你们的弟弟提醒。<br>Bad: 小明受到你们的弟弟出发。 |
| 提醒 → 昂贵 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外八位工人受到她提醒了。<br>Bad: 另外八位工人受到她昂贵了。 |
| 提醒 → 普通 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外四个上级受到郑大妈提醒了。<br>Bad: 另外四个上级受到郑大妈普通了。 |
| 提醒 → 沉默 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这位顾客被小明提醒了。<br>Bad: 这位顾客被小明沉默了。 |
| 提醒 → 粗旷 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈的朋友被小王提醒了。<br>Bad: 郑大妈的朋友被小王粗旷了。 |
| 提醒 → 精致 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈的妈妈受到他们提醒了。<br>Bad: 周大妈的妈妈受到他们精致了。 |
| 提醒 → 苦恼 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶的老板遭到她提醒了。<br>Bad: 张婶的老板遭到她苦恼了。 |
| 提醒 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生受到你们的妹妹提醒。<br>Bad: 王先生受到你们的妹妹闲逛。 |
| 支持 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王五受到我们的同事支持。<br>Bad: 王五受到我们的同事偷听。 |
| 支持 → 叹息 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥受到张婶的兄弟支持。<br>Bad: 杨大哥受到张婶的兄弟叹息。 |
| 支持 → 品茶 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 刘先生被张婶的上级支持。<br>Bad: 刘先生被张婶的上级品茶。 |
| 支持 → 爬行 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太受到另外九位音乐家支持。<br>Bad: 吴太太受到另外九位音乐家爬行。 |
| 支持 → 玩耍 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈受到那三个女儿支持。<br>Bad: 周大妈受到那三个女儿玩耍。 |
| 支持 → 躺下 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王姨受到那三个司机支持。<br>Bad: 王姨受到那三个司机躺下。 |
| 支持 → 过去 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明被你们的哥哥支持。<br>Bad: 小明被你们的哥哥过去。 |
| 支持 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥被那个领导支持。<br>Bad: 冯大哥被那个领导闲逛。 |
| 教育 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘受到另外两位员工教育。<br>Bad: 王大娘受到另外两位员工启程。 |
| 教育 → 品茶 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈遭到那四位演奏员教育。<br>Bad: 周大妈遭到那四位演奏员品茶。 |
| 教育 → 失望 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那四个司机遭到我教育了。<br>Bad: 那四个司机遭到我失望了。 |
| 教育 → 清淡 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这两位上级遭到李先生教育了。<br>Bad: 这两位上级遭到李先生清淡了。 |
| 教育 → 走路 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐受到她的父亲教育。<br>Bad: 徐小姐受到她的父亲走路。 |
| 欺负 → 昂贵 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个女儿被你欺负了。<br>Bad: 这个女儿被你昂贵了。 |
| 欺负 → 欢快 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个奴隶被周大妈欺负了。<br>Bad: 那个奴隶被周大妈欢快了。 |
| 欺负 → 激昂 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们的姐妹遭到冯大哥欺负了。<br>Bad: 他们的姐妹遭到冯大哥激昂了。 |
| 欺负 → 爬行 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王遭到他们的母亲欺负。<br>Bad: 小王遭到他们的母亲爬行。 |
| 欺负 → 跳舞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太遭到周大妈的老板欺负。<br>Bad: 吴太太遭到周大妈的老板跳舞。 |
| 欺负 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷被她的上级欺负。<br>Bad: 胡大爷被她的上级运动。 |
| 欺负 → 难过 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位打工人被我欺负了。<br>Bad: 那位打工人被我难过了。 |
| 欺负 → 高兴 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一位吉他手受到吴太太欺负了。<br>Bad: 这一位吉他手受到吴太太高兴了。 |
| 欺负 → 鲜美 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一位舞者遭到王姨欺负了。<br>Bad: 另外一位舞者遭到王姨鲜美了。 |
| 欺骗 → 愤怒 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三的领导被他们欺骗了。<br>Bad: 张三的领导被他们愤怒了。 |
| 欺骗 → 方便 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太的儿子受到宋女士欺骗了。<br>Bad: 李太太的儿子受到宋女士方便了。 |
| 欺骗 → 沉默 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个老板遭到吴太太欺骗了。<br>Bad: 那个老板遭到吴太太沉默了。 |
| 欺骗 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐遭到那九个演员欺骗。<br>Bad: 陈大姐遭到那九个演员起飞。 |
| 欺骗 → 醇厚 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 徐小姐的母亲遭到小王欺骗了。<br>Bad: 徐小姐的母亲遭到小王醇厚了。 |
| 照顾 → 伤心 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 这两位父亲受到我照顾了。<br>Bad: 这两位父亲受到我伤心了。 |
| 照顾 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生受到另外一个罪犯照顾。<br>Bad: 李先生受到另外一个罪犯偷听。 |
| 照顾 → 入睡 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太受到另外两个哥哥照顾。<br>Bad: 吴太太受到另外两个哥哥入睡。 |
| 照顾 → 普通 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位音乐家受到我照顾了。<br>Bad: 那位音乐家受到我普通了。 |
| 照顾 → 有趣 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三位演员被郑大妈照顾了。<br>Bad: 这三位演员被郑大妈有趣了。 |
| 照顾 → 浅显 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个朋友被你照顾了。<br>Bad: 那个朋友被你浅显了。 |
| 照顾 → 激昂 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 那位老师受到张婶照顾了。<br>Bad: 那位老师受到张婶激昂了。 |
| 照顾 → 玩耍 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷受到这个消费者照顾。<br>Bad: 胡大爷受到这个消费者玩耍。 |
| 照顾 → 苦恼 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太的老板被你们照顾了。<br>Bad: 李太太的老板被你们苦恼了。 |
| 照顾 → 酥脆 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那四位父亲受到她们照顾了。<br>Bad: 那四位父亲受到她们酥脆了。 |
| 照顾 → 鲜美 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这五个司机受到杨大哥照顾了。<br>Bad: 这五个司机受到杨大哥鲜美了。 |
| 爱戴 → 出发 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王大娘被那位父亲爱戴。<br>Bad: 王大娘被那位父亲出发。 |
| 爱戴 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生受到这个儿子爱戴。<br>Bad: 李先生受到这个儿子启程。 |
| 爱戴 → 品茶 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太被那位老师爱戴。<br>Bad: 李太太被那位老师品茶。 |
| 爱戴 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨受到这四位空姐爱戴。<br>Bad: 王姨受到这四位空姐唱歌。 |
| 爱戴 → 运动 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷受到这个领导爱戴。<br>Bad: 胡大爷受到这个领导运动。 |
| 爱戴 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王先生受到这个弟弟爱戴。<br>Bad: 王先生受到这个弟弟闲逛。 |
| 爱护 → 启程 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶受到那个妹妹爱护。<br>Bad: 张婶受到那个妹妹启程。 |
| 爱护 → 打架 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太受到另外九位老板爱护。<br>Bad: 李太太受到另外九位老板打架。 |
| 爱护 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐受到那位母亲爱护。<br>Bad: 王小姐受到那位母亲游泳。 |
| 爱护 → 看戏 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈被这位消费者爱护。<br>Bad: 郑大妈被这位消费者看戏。 |
| 爱护 → 过去 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥被那九位员工爱护。<br>Bad: 冯大哥被那九位员工过去。 |
| 眼睛 → 桌子 | passive_body_part | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他们被刘先生检查了眼睛。<br>Bad: 他们被刘先生检查了桌子。 |
| 眼睛 → 裤子 | passive_body_part | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷被吴太太检查了眼睛。<br>Bad: 胡大爷被吴太太检查了裤子。 |
| 称赞 → 入睡 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 周大妈受到我的女儿称赞。<br>Bad: 周大妈受到我的女儿入睡。 |
| 称赞 → 唱歌 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷被刘先生的弟弟称赞。<br>Bad: 胡大爷被刘先生的弟弟唱歌。 |
| 称赞 → 坎坷 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我们的姐姐被李先生称赞了。<br>Bad: 我们的姐姐被李先生坎坷了。 |
| 称赞 → 年轻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那两个顾客受到王五称赞了。<br>Bad: 那两个顾客受到王五年轻了。 |
| 称赞 → 欢快 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那六位母亲被郑大妈称赞了。<br>Bad: 那六位母亲被郑大妈欢快了。 |
| 称赞 → 精致 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这三位舞者被胡大爷称赞了。<br>Bad: 这三位舞者被胡大爷精致了。 |
| 称赞 → 酥软 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太的领导被刘先生称赞了。<br>Bad: 李太太的领导被刘先生酥软了。 |
| 约束 → 健身 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐被小明的兄弟约束。<br>Bad: 陈大姐被小明的兄弟健身。 |
| 约束 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张三被你们的领导约束。<br>Bad: 张三被你们的领导偷听。 |
| 约束 → 呼吸 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生遭到那个顾客约束。<br>Bad: 刘先生遭到那个顾客呼吸。 |
| 约束 → 辛辣 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我的老板被他约束了。<br>Bad: 我的老板被他辛辣了。 |
| 维护 → 游泳 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三受到他们的爸爸维护。<br>Bad: 张三受到他们的爸爸游泳。 |
| 耳朵 → 椅子 | passive_body_part | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你被杨大哥包扎了耳朵。<br>Bad: 你被杨大哥包扎了椅子。 |
| 表扬 → 便宜 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个女儿受到他们表扬了。<br>Bad: 这个女儿受到他们便宜了。 |
| 表扬 → 困惑 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个罪犯被冯大哥表扬了。<br>Bad: 这个罪犯被冯大哥困惑了。 |
| 表扬 → 宁静 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个顾客受到小王表扬了。<br>Bad: 那个顾客受到小王宁静了。 |
| 表扬 → 忧郁 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那九位服务员受到李太太表扬了。<br>Bad: 那九位服务员受到李太太忧郁了。 |
| 表扬 → 快乐 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一位钢琴家被郑大妈表扬了。<br>Bad: 这一位钢琴家被郑大妈快乐了。 |
| 表扬 → 悲伤 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位下属受到你们表扬了。<br>Bad: 这位下属受到你们悲伤了。 |
| 表扬 → 有趣 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一个奴隶受到我们表扬了。<br>Bad: 这一个奴隶受到我们有趣了。 |
| 表扬 → 欢快 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六个司机被李太太表扬了。<br>Bad: 这六个司机被李太太欢快了。 |
| 表扬 → 沉默 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外七个顾客受到小明表扬了。<br>Bad: 另外七个顾客受到小明沉默了。 |
| 表扬 → 深刻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这七位顾客被张先生表扬了。<br>Bad: 这七位顾客被张先生深刻了。 |
| 表扬 → 滑嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们的兄弟被你表扬了。<br>Bad: 你们的兄弟被你滑嫩了。 |
| 表扬 → 热烈 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位父亲被张婶表扬了。<br>Bad: 这位父亲被张婶热烈了。 |
| 表扬 → 爬行 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太受到他们的妈妈表扬。<br>Bad: 吴太太受到他们的妈妈爬行。 |
| 表扬 → 跳舞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生受到另外三个记者表扬。<br>Bad: 李先生受到另外三个记者跳舞。 |
| 表扬 → 辛辣 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一个打工人受到小明表扬了。<br>Bad: 这一个打工人受到小明辛辣了。 |
| 表扬 → 酥脆 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥的朋友受到李先生表扬了。<br>Bad: 冯大哥的朋友受到李先生酥脆了。 |
| 表扬 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外一个领导被她们表扬了。<br>Bad: 另外一个领导被她们酸爽了。 |
| 表扬 → 颤抖 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生受到小明的上级表扬。<br>Bad: 张先生受到小明的上级颤抖。 |
| 表扬 → 高昂 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外九位空姐被胡大爷表扬了。<br>Bad: 另外九位空姐被胡大爷高昂了。 |
| 诽谤 → 低沉 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这四个女儿遭到张夫人诽谤了。<br>Bad: 这四个女儿遭到张夫人低沉了。 |
| 诽谤 → 保守 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 另外五个朋友受到我们诽谤了。<br>Bad: 另外五个朋友受到我们保守了。 |
| 诽谤 → 偷听 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李先生受到那个女儿诽谤。<br>Bad: 李先生受到那个女儿偷听。 |
| 诽谤 → 呼吸 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨被她的妈妈诽谤。<br>Bad: 王姨被她的妈妈呼吸。 |
| 诽谤 → 失望 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位母亲受到你诽谤了。<br>Bad: 那位母亲受到你失望了。 |
| 诽谤 → 方便 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位父亲遭到她们诽谤了。<br>Bad: 那位父亲遭到她们方便了。 |
| 诽谤 → 有趣 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个钢琴家受到她们诽谤了。<br>Bad: 那个钢琴家受到她们有趣了。 |
| 诽谤 → 特殊 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个吉他手遭到杨大哥诽谤了。<br>Bad: 这个吉他手遭到杨大哥特殊了。 |
| 诽谤 → 走路 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥受到另外四位演奏员诽谤。<br>Bad: 杨大哥受到另外四位演奏员走路。 |
| 诽谤 → 难过 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个朋友遭到他诽谤了。<br>Bad: 那个朋友遭到他难过了。 |
| 责备 → 偷听 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨遭到那八个妹妹责备。<br>Bad: 王姨遭到那八个妹妹偷听。 |
| 责备 → 冷静 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这九位老板被张先生责备了。<br>Bad: 这九位老板被张先生冷静了。 |
| 责备 → 凶猛 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太的学生受到赵大爷责备了。<br>Bad: 李太太的学生受到赵大爷凶猛了。 |
| 责备 → 成熟 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张三的哥哥遭到郑大妈责备了。<br>Bad: 张三的哥哥遭到郑大妈成熟了。 |
| 责备 → 深情 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个小孩受到冯大哥责备了。<br>Bad: 这个小孩受到冯大哥深情了。 |
| 责备 → 激昂 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这位记者遭到陈大姐责备了。<br>Bad: 这位记者遭到陈大姐激昂了。 |
| 责备 → 酥软 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生的母亲遭到他责备了。<br>Bad: 张先生的母亲遭到他酥软了。 |
| 赞赏 → 过来 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四受到那位工人赞赏。<br>Bad: 李四受到那位工人过来。 |
| 赞赏 → 颤抖 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨被这位服务员赞赏。<br>Bad: 王姨被这位服务员颤抖。 |
| 辩护 → 打架 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王被这三个儿子辩护。<br>Bad: 小王被这三个儿子打架。 |
| 辩护 → 起飞 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太被这个女儿辩护。<br>Bad: 李太太被这个女儿起飞。 |
| 辩护 → 过去 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐被她的上级辩护。<br>Bad: 陈大姐被她的上级过去。 |
| 辩护 → 闲逛 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥被这四位员工辩护。<br>Bad: 冯大哥被这四位员工闲逛。 |
| 追捧 → 叹息 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 赵大爷受到另外四个妹妹追捧。<br>Bad: 赵大爷受到另外四个妹妹叹息。 |
| 追捧 → 走路 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太被这八个小孩追捧。<br>Bad: 李太太被这八个小孩走路。 |
| 青睐 → 听课 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶受到那两位钢琴家青睐。<br>Bad: 张婶受到那两位钢琴家听课。 |
| 青睐 → 打架 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小明被他们的员工青睐。<br>Bad: 小明被他们的员工打架。 |
| 青睐 → 睡觉 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐受到另外三个弟弟青睐。<br>Bad: 陈大姐受到另外三个弟弟睡觉。 |
| 青睐 → 站立 | passive_intransitive | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈受到这六位音乐家青睐。<br>Bad: 周大妈受到这六位音乐家站立。 |
| 青睐 → 走路 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷被另外七位吉他手青睐。<br>Bad: 赵大爷被另外七位吉他手走路。 |
| 鼓励 → 保守 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那位司机受到张婶鼓励了。<br>Bad: 那位司机受到张婶保守了。 |
| 鼓励 → 动听 | passive_no_adj | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐的兄弟被你们鼓励了。<br>Bad: 陈大姐的兄弟被你们动听了。 |
| 鼓励 → 听课 | passive_intransitive | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王被胡大爷的领导鼓励。<br>Bad: 小王被胡大爷的领导听课。 |
| 鼓励 → 年轻 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们的老师受到吴太太鼓励了。<br>Bad: 她们的老师受到吴太太年轻了。 |
| 鼓励 → 精致 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个工人受到我们鼓励了。<br>Bad: 这个工人受到我们精致了。 |
| 鼓励 → 酥脆 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个下属受到王姨鼓励了。<br>Bad: 那个下属受到王姨酥脆了。 |
| 鼓励 → 酥软 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 那个工人被李四鼓励了。<br>Bad: 那个工人被李四酥软了。 |
| 鼓励 → 酸爽 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这一位打工人受到李四鼓励了。<br>Bad: 这一位打工人受到李四酸爽了。 |
| 鼓励 → 鲜嫩 | passive_no_adj | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个司机受到杨大哥鼓励了。<br>Bad: 这个司机受到杨大哥鲜嫩了。 |

## quantifiers

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad inserts 没; bad deletes 的 | superlative_quantifiers_2 | 300 | 0.5133 | 0.8800 | -0.3667 | 0.0000 | Good: 有的警察制作了最多一本手账。<br>Bad: 没有警察制作了最多一本手账。 |
| 超过 → 至少 | superlative_quantifiers_1 | 300 | 0.0167 | 0.1267 | -0.1100 | 0.0000 | Good: 没有学生看了超过八本书。<br>Bad: 没有学生看了至少八本书。 |

## question

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad deletes 唱歌; bad inserts 唱歌 | question_A_not_A_daodi_a | 9 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小王到底唱歌不唱歌？<br>Bad: 小王到底不唱歌唱歌？ |
| multiple edits: bad deletes 跑步; bad inserts 跑步 | question_A_not_A_daodi_a | 9 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你到底跑步不跑步？<br>Bad: 你到底不跑步跑步？ |
| multiple edits: bad deletes 健身; bad inserts 健身 | question_A_not_A_daodi_a | 8 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥到底健身不健身？<br>Bad: 杨大哥到底不健身健身？ |
| multiple edits: bad deletes 跑步; bad inserts 跑步 | question_A_not_A_daodi_b | 8 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你到底跑步不跑步？<br>Bad: 你到底不跑步跑步？ |
| multiple edits: bad deletes 听课; bad inserts 听课 | question_A_not_A_daodi_b | 7 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张三到底听课不听课？<br>Bad: 张三到底不听课听课？ |
| multiple edits: bad deletes 唱歌; bad inserts 唱歌 | question_A_not_A_daodi_b | 7 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 张夫人到底唱歌不唱歌？<br>Bad: 张夫人到底不唱歌唱歌？ |
| multiple edits: bad deletes 偷听; bad inserts 偷听 | question_A_not_A_daodi_b | 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小明到底偷听不偷听？<br>Bad: 小明到底不偷听偷听？ |
| multiple edits: bad deletes 听课; bad inserts 听课 | question_A_not_A_daodi_a | 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 郑大妈到底听课不听课？<br>Bad: 郑大妈到底不听课听课？ |
| multiple edits: bad deletes 过来; bad inserts 过来 | question_A_not_A_daodi_b | 6 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 吴太太到底过来不过来？<br>Bad: 吴太太到底不过来过来？ |
| multiple edits: bad deletes 熔化; bad inserts 熔化 | question_A_not_A_daodi_b | 5 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 小明到底熔化不熔化？<br>Bad: 小明到底不熔化熔化？ |
| 难道 → 到底 | question_particle_nandao | 300 | 0.9900 | 0.0033 | +0.9867 | 0.0000 | Good: 我们现在难道已经唱过美声了吗？<br>Bad: 我们现在到底已经唱过美声了吗？ |
| 到底 → 难道 | question_daodi_nandao_A_not_A_intran | 300 | 0.0300 | 0.9533 | -0.9233 | 0.0000 | Good: 王五到底爬行不爬行？<br>Bad: 王五难道爬行不爬行？ |
| 到底 → 难道 | question_daodi_nandao_2 | 300 | 0.0300 | 0.9433 | -0.9133 | 0.0000 | Good: 他们到底包扎了这些手没？<br>Bad: 他们难道包扎了这些手没？ |
| multiple edits: bad deletes 玩耍; bad inserts 玩耍 | question_A_not_A_daodi_a | 11 | 0.0909 | 1.0000 | -0.9091 | 0.0000 | Good: 小明到底玩耍不玩耍？<br>Bad: 小明到底不玩耍玩耍？ |
| multiple edits: bad deletes 坐下; bad inserts 坐下 | question_A_not_A_daodi_a | 10 | 0.9000 | 0.0000 | +0.9000 | 0.0000 | Good: 你到底坐下不坐下？<br>Bad: 你到底不坐下坐下？ |
| multiple edits: bad deletes 过来; bad inserts 过来 | question_A_not_A_daodi_a | 10 | 0.1000 | 1.0000 | -0.9000 | 0.0000 | Good: 她到底过来不过来？<br>Bad: 她到底不过来过来？ |
| multiple edits: bad deletes 停下; bad inserts 停下 | question_A_not_A_daodi_b | 9 | 1.0000 | 0.1111 | +0.8889 | 0.0000 | Good: 吴太太到底停下不停下？<br>Bad: 吴太太到底不停下停下？ |
| multiple edits: bad deletes 健身; bad inserts 健身 | question_A_not_A_daodi_b | 7 | 1.0000 | 0.1429 | +0.8571 | 0.0000 | Good: 徐小姐到底健身不健身？<br>Bad: 徐小姐到底不健身健身？ |
| multiple edits: bad deletes 玩耍; bad inserts 玩耍 | question_A_not_A_daodi_b | 7 | 0.1429 | 1.0000 | -0.8571 | 0.0000 | Good: 小王到底玩耍不玩耍？<br>Bad: 小王到底不玩耍玩耍？ |
| multiple edits: bad deletes 站立; bad inserts 站立 | question_A_not_A_daodi_a | 7 | 0.1429 | 1.0000 | -0.8571 | 0.0000 | Good: 她们到底站立不站立？<br>Bad: 她们到底不站立站立？ |
| multiple edits: bad deletes 停下; bad inserts 停下 | question_A_not_A_daodi_a | 7 | 0.8571 | 0.0000 | +0.8571 | 0.0000 | Good: 冯大哥到底停下不停下？<br>Bad: 冯大哥到底不停下停下？ |
| 到底 → 难道 | question_daodi_nandao_A_not_A_tran | 300 | 0.0733 | 0.9133 | -0.8400 | 0.0000 | Good: 她到底清洗不清洗杯子？<br>Bad: 她难道清洗不清洗杯子？ |
| multiple edits: bad deletes 坐下; bad inserts 坐下 | question_A_not_A_daodi_b | 6 | 1.0000 | 0.1667 | +0.8333 | 0.0000 | Good: 张婶到底坐下不坐下？<br>Bad: 张婶到底不坐下坐下？ |
| multiple edits: bad deletes 站立; bad inserts 站立 | question_A_not_A_daodi_b | 6 | 0.1667 | 1.0000 | -0.8333 | 0.0000 | Good: 他到底站立不站立？<br>Bad: 他到底不站立站立？ |
| multiple edits: bad deletes 躺下; bad inserts 躺下 | question_A_not_A_daodi_a | 6 | 0.1667 | 1.0000 | -0.8333 | 0.0000 | Good: 她们到底躺下不躺下？<br>Bad: 她们到底不躺下躺下？ |
| 呢 → 吗 | question_particle_daodi_choice_intran | 300 | 0.9567 | 0.1300 | +0.8267 | 0.0000 | Good: 你们到底想跑步还是想微笑呢？<br>Bad: 你们到底想跑步还是想微笑吗？ |
| multiple edits: bad deletes 叹息; bad inserts 叹息 | question_A_not_A_daodi_a | 10 | 0.9000 | 0.1000 | +0.8000 | 0.0000 | Good: 张三到底叹息不叹息？<br>Bad: 张三到底不叹息叹息？ |
| multiple edits: bad deletes 闲逛; bad inserts 闲逛 | question_A_not_A_daodi_a | 9 | 0.2222 | 1.0000 | -0.7778 | 0.0000 | Good: 宋女士到底闲逛不闲逛？<br>Bad: 宋女士到底不闲逛闲逛？ |
| multiple edits: bad deletes 变质; bad inserts 变质 | question_A_not_A_daodi_b | 7 | 1.0000 | 0.2857 | +0.7143 | 0.0000 | Good: 张婶到底变质不变质？<br>Bad: 张婶到底不变质变质？ |
| multiple edits: bad deletes 出发; bad inserts 出发 | question_A_not_A_daodi_a | 9 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 她们到底出发不出发？<br>Bad: 她们到底不出发出发？ |
| bad inserts 张三 | question_A_not_A | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 张三领养小狗不领养小狗？<br>Bad: 张三领养小狗张三不领养小狗？ |
| bad inserts 李四 | question_A_not_A | 3 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 李四包扎肚子不包扎肚子？<br>Bad: 李四包扎肚子李四不包扎肚子？ |
| 呢 → 吗 | question_particle_daodi_choice_tran | 300 | 0.8633 | 0.2167 | +0.6467 | 0.0000 | Good: 王小姐到底是想检查手还是想炖鱼呢？<br>Bad: 王小姐到底是想检查手还是想炖鱼吗？ |
| multiple edits: bad deletes 出发; bad inserts 出发 | question_A_not_A_daodi_b | 8 | 1.0000 | 0.3750 | +0.6250 | 0.0000 | Good: 张三到底出发不出发？<br>Bad: 张三到底不出发出发？ |
| multiple edits: bad inserts 不; 不愿意 -> 想 | question_nandao_negation | 30 | 0.1667 | 0.7667 | -0.6000 | 0.0000 | Good: 李先生难道不愿意吹双簧吗？<br>Bad: 李先生不难道想吹双簧吗？ |
| multiple edits: bad deletes 叹息; bad inserts 叹息 | question_A_not_A_daodi_b | 5 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 张夫人到底叹息不叹息？<br>Bad: 张夫人到底不叹息叹息？ |
| multiple edits: bad inserts 不; bad deletes 不 | question_nandao_negation | 99 | 0.3737 | 0.8889 | -0.5152 | 0.0000 | Good: 这八位消费者难道不愿意厌恶音乐家吗？<br>Bad: 这八位消费者不难道愿意厌恶音乐家吗？ |
| bad inserts 她 | question_A_not_A | 14 | 0.2857 | 0.7857 | -0.5000 | 0.0000 | Good: 她包扎手不包扎手？<br>Bad: 她包扎手她不包扎手？ |
| multiple edits: bad deletes 难道; bad inserts 难道 | question_nandao_scope_1 | 300 | 0.0900 | 0.5500 | -0.4600 | 0.0000 | Good: 难道王先生认为李先生可能嫌弃顾客吗？<br>Bad: 王先生认为李先生难道可能嫌弃顾客吗？ |
| multiple edits: bad deletes 溜走; bad inserts 溜走 | question_A_not_A_daodi_a | 11 | 0.5455 | 1.0000 | -0.4545 | 0.0000 | Good: 胡大爷到底溜走不溜走？<br>Bad: 胡大爷到底不溜走溜走？ |
| multiple edits: bad deletes 睡觉; bad inserts 睡觉 | question_A_not_A_daodi_b | 7 | 1.0000 | 0.5714 | +0.4286 | 0.0000 | Good: 小明到底睡觉不睡觉？<br>Bad: 小明到底不睡觉睡觉？ |
| multiple edits: bad deletes 闲逛; bad inserts 闲逛 | question_A_not_A_daodi_b | 7 | 0.5714 | 1.0000 | -0.4286 | 0.0000 | Good: 他到底闲逛不闲逛？<br>Bad: 他到底不闲逛闲逛？ |
| multiple edits: bad deletes 爬行; bad inserts 爬行 | question_A_not_A_daodi_b | 7 | 0.1429 | 0.5714 | -0.4286 | 0.0000 | Good: 她到底爬行不爬行？<br>Bad: 她到底不爬行爬行？ |
| multiple edits: bad deletes 不从; bad inserts 不从 | question_V_not_VP_1 | 242 | 0.5124 | 0.9050 | -0.3926 | 0.0000 | Good: 小王的母亲今天从不从火山出发？<br>Bad: 小王的母亲今天从火山出发不从？ |
| multiple edits: bad deletes 来; bad inserts 来 | question_A_not_A_daodi_a | 11 | 1.0000 | 0.6364 | +0.3636 | 0.0000 | Good: 张先生到底来不来？<br>Bad: 张先生到底不来来？ |
| multiple edits: bad inserts 不; 不愿意 -> 希望 | question_nandao_negation | 40 | 0.5500 | 0.9000 | -0.3500 | 0.0000 | Good: 你难道不愿意清洗杯子吗？<br>Bad: 你不难道希望清洗杯子吗？ |
| multiple edits: bad deletes 碎掉; bad inserts 碎掉 | question_A_not_A_daodi_b | 6 | 0.5000 | 0.8333 | -0.3333 | 0.0000 | Good: 你们到底碎掉不碎掉？<br>Bad: 你们到底不碎掉碎掉？ |
| bad inserts 王姨 | question_A_not_A | 6 | 0.0000 | 0.3333 | -0.3333 | 0.0000 | Good: 王姨创作小说不创作小说？<br>Bad: 王姨创作小说王姨不创作小说？ |
| multiple edits: bad deletes 受潮; bad inserts 受潮 | question_A_not_A_daodi_b | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 何太太到底受潮不受潮？<br>Bad: 何太太到底不受潮受潮？ |
| multiple edits: bad deletes 起飞; bad inserts 起飞 | question_A_not_A_daodi_b | 7 | 0.7143 | 1.0000 | -0.2857 | 0.0000 | Good: 他们到底起飞不起飞？<br>Bad: 他们到底不起飞起飞？ |
| multiple edits: bad inserts 不; 不想 -> 愿意 | question_nandao_negation | 25 | 0.7200 | 1.0000 | -0.2800 | 0.0000 | Good: 他们难道不想来吗？<br>Bad: 他们不难道愿意来吗？ |
| multiple edits: bad deletes 从不; bad inserts 不从 | question_V_not_VP_1 | 58 | 0.6552 | 0.9310 | -0.2759 | 0.0000 | Good: 你今天从不从沙漠出发？<br>Bad: 你今天从沙漠出发不从？ |
| multiple edits: bad deletes 难道; bad inserts 难道 | question_nandao_raising_2 | 162 | 1.0000 | 0.7469 | +0.2531 | 0.0000 | Good: 难道有打工人不游泳吗？<br>Bad: 有打工人难道不游泳吗？ |
| bad inserts 周大妈 | question_A_not_A | 8 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 周大妈制作电影不制作电影？<br>Bad: 周大妈制作电影周大妈不制作电影？ |
| multiple edits: bad deletes 来; bad inserts 来 | question_A_not_A_daodi_b | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 你们到底来不来？<br>Bad: 你们到底不来来？ |
| multiple edits: bad deletes 躺下; bad inserts 躺下 | question_A_not_A_daodi_b | 8 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 她们到底躺下不躺下？<br>Bad: 她们到底不躺下躺下？ |
| multiple edits: bad deletes 睡觉; bad inserts 睡觉 | question_A_not_A_daodi_a | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 你们到底睡觉不睡觉？<br>Bad: 你们到底不睡觉睡觉？ |
| multiple edits: bad inserts 是我; bad deletes 是我 | question_nandao_raising_1_b | 21 | 1.0000 | 0.7619 | +0.2381 | 0.0000 | Good: 难道是我先清蒸的吗？<br>Bad: 是我难道先清蒸的吗？ |
| 难道 → 到底 | question_daodi_nandao_1 | 300 | 1.0000 | 0.7733 | +0.2267 | 0.0000 | Good: 她难道不捕捉蛇吗？<br>Bad: 她到底不捕捉蛇吗？ |
| bad inserts 刘先生 | question_A_not_A | 9 | 0.0000 | 0.2222 | -0.2222 | 0.0000 | Good: 刘先生卖开瓶器不卖开瓶器？<br>Bad: 刘先生卖开瓶器刘先生不卖开瓶器？ |
| multiple edits: bad deletes 腐烂; bad inserts 腐烂 | question_A_not_A_daodi_b | 9 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 张先生到底腐烂不腐烂？<br>Bad: 张先生到底不腐烂腐烂？ |
| multiple edits: bad deletes 把不; bad inserts 不把 | question_V_not_VP_2 | 300 | 0.8233 | 0.6067 | +0.2167 | 0.0000 | Good: 小王把不把那本教材递给另外四个弟弟？<br>Bad: 小王把那本教材递给另外四个弟弟不把？ |
| bad inserts 小明 | question_A_not_A | 5 | 0.0000 | 0.2000 | -0.2000 | 0.0000 | Good: 小明开卡车不开卡车？<br>Bad: 小明开卡车小明不开卡车？ |
| bad inserts 李先生 | question_A_not_A | 5 | 0.0000 | 0.2000 | -0.2000 | 0.0000 | Good: 李先生清洗杯子不清洗杯子？<br>Bad: 李先生清洗杯子李先生不清洗杯子？ |
| multiple edits: bad deletes 溜走; bad inserts 溜走 | question_A_not_A_daodi_b | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 他们到底溜走不溜走？<br>Bad: 他们到底不溜走溜走？ |
| multiple edits: bad inserts 确定; bad deletes 确定 | question_nandao_scope_2 | 98 | 0.3571 | 0.1633 | +0.1939 | 0.0000 | Good: 你难道确定这十位父亲拉小提琴吗？<br>Bad: 你确定难道这十位父亲拉小提琴吗？ |
| bad inserts 他 | question_A_not_A | 22 | 1.0000 | 0.8182 | +0.1818 | 0.0000 | Good: 他拉大提琴不拉大提琴？<br>Bad: 他拉大提琴他不拉大提琴？ |
| bad inserts 张夫人 | question_A_not_A | 6 | 0.1667 | 0.0000 | +0.1667 | 0.0000 | Good: 张夫人炖鸡不炖鸡？<br>Bad: 张夫人炖鸡张夫人不炖鸡？ |
| bad inserts 王五 | question_A_not_A | 6 | 0.0000 | 0.1667 | -0.1667 | 0.0000 | Good: 王五清洗杯子不清洗杯子？<br>Bad: 王五清洗杯子王五不清洗杯子？ |
| multiple edits: bad deletes 颤抖; bad inserts 颤抖 | question_A_not_A_daodi_a | 6 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 他们到底颤抖不颤抖？<br>Bad: 他们到底不颤抖颤抖？ |
| bad inserts 她们 | question_A_not_A | 13 | 0.5385 | 0.3846 | +0.1538 | 0.0000 | Good: 她们看录像带不看录像带？<br>Bad: 她们看录像带她们不看录像带？ |
| multiple edits: bad inserts 不; bad deletes 不 | question_daodi_negation | 300 | 0.9967 | 0.8467 | +0.1500 | 0.0000 | Good: 那六个吉他手到底不希望安慰什么？<br>Bad: 那六个吉他手不到底希望安慰什么？ |
| multiple edits: bad inserts 不; 不希望 -> 想 | question_nandao_negation | 40 | 0.7250 | 0.5750 | +0.1500 | 0.0000 | Good: 那十个司机难道不希望拍摄电影吗？<br>Bad: 那十个司机不难道想拍摄电影吗？ |
| multiple edits: bad deletes 过期; bad inserts 过期 | question_A_not_A_daodi_b | 7 | 0.1429 | 0.0000 | +0.1429 | 0.0000 | Good: 他到底过期不过期？<br>Bad: 他到底不过期过期？ |
| bad inserts 你 | question_A_not_A | 16 | 0.2500 | 0.1250 | +0.1250 | 0.0000 | Good: 你屠宰牛不屠宰牛？<br>Bad: 你屠宰牛你不屠宰牛？ |
| bad inserts 王先生 | question_A_not_A | 8 | 0.0000 | 0.1250 | -0.1250 | 0.0000 | Good: 王先生唱戏曲不唱戏曲？<br>Bad: 王先生唱戏曲王先生不唱戏曲？ |
| multiple edits: bad deletes 颤抖; bad inserts 颤抖 | question_A_not_A_daodi_b | 8 | 0.8750 | 1.0000 | -0.1250 | 0.0000 | Good: 他们到底颤抖不颤抖？<br>Bad: 他们到底不颤抖颤抖？ |
| multiple edits: bad inserts 是; bad deletes 是 | question_nandao_raising_3 | 300 | 0.6333 | 0.7567 | -0.1233 | 0.0000 | Good: 你们难道是一周前才起飞的吗？<br>Bad: 你们是难道一周前才起飞的吗？ |
| multiple edits: bad inserts 不; 不想 -> 希望 | question_nandao_negation | 27 | 0.7778 | 0.8889 | -0.1111 | 0.0000 | Good: 那两位工人难道不想拍摄记录片吗？<br>Bad: 那两位工人不难道希望拍摄记录片吗？ |
| 。 → 呢？ | question_A_not_A_indirect | 300 | 0.0000 | 0.1067 | -0.1067 | 0.0000 | Good: 李四想讨论她们跑步不跑步的问题。<br>Bad: 李四想讨论她们跑步不跑步的问题呢？ |
| bad inserts 我 | question_A_not_A | 19 | 0.0000 | 0.1053 | -0.1053 | 0.0000 | Good: 我喝啤酒不喝啤酒？<br>Bad: 我喝啤酒我不喝啤酒？ |
| multiple edits: bad inserts 不; 不希望 -> 愿意 | question_nandao_negation | 39 | 0.9487 | 0.8462 | +0.1026 | 0.0000 | Good: 这五位老师难道不希望停下吗？<br>Bad: 这五位老师不难道愿意停下吗？ |
| multiple edits: bad deletes 起飞; bad inserts 起飞 | question_A_not_A_daodi_a | 10 | 0.9000 | 1.0000 | -0.1000 | 0.0000 | Good: 她到底起飞不起飞？<br>Bad: 她到底不起飞起飞？ |
| multiple edits: bad inserts 是他; bad deletes 是他 | question_nandao_raising_1_b | 22 | 0.9091 | 1.0000 | -0.0909 | 0.0000 | Good: 难道是他率先支持的吗？<br>Bad: 是他难道率先支持的吗？ |
| multiple edits: bad deletes 过去; bad inserts 过去 | question_A_not_A_daodi_b | 11 | 0.0000 | 0.0909 | -0.0909 | 0.0000 | Good: 她到底过去不过去？<br>Bad: 她到底不过去过去？ |
| multiple edits: bad inserts 相信; bad deletes 相信 | question_nandao_scope_2 | 104 | 0.3654 | 0.2981 | +0.0673 | 0.0000 | Good: 你难道相信那九个打工人喝橙汁吗？<br>Bad: 你相信难道那九个打工人喝橙汁吗？ |
| multiple edits: bad deletes 呼吸; bad inserts 呼吸 | question_A_not_A_daodi_a | 17 | 1.0000 | 0.9412 | +0.0588 | 0.0000 | Good: 张婶到底呼吸不呼吸？<br>Bad: 张婶到底不呼吸呼吸？ |
| multiple edits: bad deletes 难道; bad inserts 难道 | question_nandao_raising_1_b | 226 | 1.0000 | 0.9469 | +0.0531 | 0.0000 | Good: 难道是刘先生首先爱护的吗？<br>Bad: 是刘先生难道首先爱护的吗？ |
| multiple edits: bad inserts 认为; bad deletes 认为 | question_nandao_scope_2 | 98 | 0.1429 | 0.1939 | -0.0510 | 0.0000 | Good: 我难道认为这四位父亲看录像带吗？<br>Bad: 我认为难道这四位父亲看录像带吗？ |
| bad inserts 他们 | question_A_not_A | 23 | 0.4348 | 0.4783 | -0.0435 | 0.0000 | Good: 他们打碎杯子不打碎杯子？<br>Bad: 他们打碎杯子他们不打碎杯子？ |
| multiple edits: bad inserts 有人; bad deletes 有人 | question_nandao_raising_2 | 138 | 1.0000 | 0.9710 | +0.0290 | 0.0000 | Good: 难道有人原谅演员吗？<br>Bad: 有人难道原谅演员吗？ |
| multiple edits: bad inserts 是; bad deletes 是 | question_nandao_raising_1_a | 300 | 0.9767 | 0.9733 | +0.0033 | 0.0000 | Good: 难道是徐小姐先弹的吗？<br>Bad: 是难道徐小姐先弹的吗？ |
| bad inserts 我们 | question_A_not_A | 20 | 0.1500 | 0.1500 | +0.0000 | 0.0000 | Good: 我们打碎杯子不打碎杯子？<br>Bad: 我们打碎杯子我们不打碎杯子？ |
| bad inserts 你们 | question_A_not_A | 18 | 0.0556 | 0.0556 | +0.0000 | 0.0000 | Good: 你们包扎手不包扎手？<br>Bad: 你们包扎手你们不包扎手？ |
| multiple edits: bad inserts 是你; bad deletes 是你 | question_nandao_raising_1_b | 17 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 难道是你率先责备的吗？<br>Bad: 是你难道率先责备的吗？ |
| multiple edits: bad inserts 是她; bad deletes 是她 | question_nandao_raising_1_b | 14 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 难道是她首先称赞的吗？<br>Bad: 是她难道首先称赞的吗？ |
| multiple edits: bad deletes 入睡; bad inserts 入睡 | question_A_not_A_daodi_a | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们到底入睡不入睡？<br>Bad: 她们到底不入睡入睡？ |
| multiple edits: bad deletes 走; bad inserts 走 | question_A_not_A_daodi_a | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他到底走不走？<br>Bad: 他到底不走走？ |
| multiple edits: bad deletes 呼吸; bad inserts 呼吸 | question_A_not_A_daodi_b | 11 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士到底呼吸不呼吸？<br>Bad: 宋女士到底不呼吸呼吸？ |
| multiple edits: bad deletes 微笑; bad inserts 微笑 | question_A_not_A_daodi_a | 11 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他到底微笑不微笑？<br>Bad: 他到底不微笑微笑？ |
| multiple edits: bad deletes 打架; bad inserts 打架 | question_A_not_A_daodi_a | 11 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥到底打架不打架？<br>Bad: 冯大哥到底不打架打架？ |
| multiple edits: bad deletes 笑; bad inserts 笑 | question_A_not_A_daodi_a | 11 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王到底笑不笑？<br>Bad: 小王到底不笑笑？ |
| multiple edits: bad deletes 看戏; bad inserts 看戏 | question_A_not_A_daodi_b | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你到底看戏不看戏？<br>Bad: 你到底不看戏看戏？ |
| multiple edits: bad deletes 跳舞; bad inserts 跳舞 | question_A_not_A_daodi_a | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨到底跳舞不跳舞？<br>Bad: 王姨到底不跳舞跳舞？ |
| multiple edits: bad deletes 跳舞; bad inserts 跳舞 | question_A_not_A_daodi_b | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她到底跳舞不跳舞？<br>Bad: 她到底不跳舞跳舞？ |
| multiple edits: bad deletes 过去; bad inserts 过去 | question_A_not_A_daodi_a | 10 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 她们到底过去不过去？<br>Bad: 她们到底不过去过去？ |
| multiple edits: bad deletes 走路; bad inserts 走路 | question_A_not_A_daodi_a | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们到底走路不走路？<br>Bad: 她们到底不走路走路？ |
| multiple edits: bad deletes 走路; bad inserts 走路 | question_A_not_A_daodi_b | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明到底走路不走路？<br>Bad: 小明到底不走路走路？ |
| bad inserts 张婶 | question_A_not_A | 8 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张婶打碎杯子不打碎杯子？<br>Bad: 张婶打碎杯子张婶不打碎杯子？ |
| bad inserts 赵大爷 | question_A_not_A | 8 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 赵大爷搬桌子不搬桌子？<br>Bad: 赵大爷搬桌子赵大爷不搬桌子？ |
| bad inserts 郑大妈 | question_A_not_A | 8 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 郑大妈开卡车不开卡车？<br>Bad: 郑大妈开卡车郑大妈不开卡车？ |
| multiple edits: bad deletes 启程; bad inserts 启程 | question_A_not_A_daodi_a | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们到底启程不启程？<br>Bad: 我们到底不启程启程？ |
| multiple edits: bad deletes 启程; bad inserts 启程 | question_A_not_A_daodi_b | 8 | 0.8750 | 0.8750 | +0.0000 | 0.0000 | Good: 陈大姐到底启程不启程？<br>Bad: 陈大姐到底不启程启程？ |
| multiple edits: bad deletes 品茶; bad inserts 品茶 | question_A_not_A_daodi_b | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我到底品茶不品茶？<br>Bad: 我到底不品茶品茶？ |
| bad inserts 何太太 | question_A_not_A | 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 何太太喝白酒不喝白酒？<br>Bad: 何太太喝白酒何太太不喝白酒？ |
| bad inserts 冯大哥 | question_A_not_A | 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 冯大哥打断脚不打断脚？<br>Bad: 冯大哥打断脚冯大哥不打断脚？ |
| bad inserts 徐小姐 | question_A_not_A | 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 徐小姐捕捉老虎不捕捉老虎？<br>Bad: 徐小姐捕捉老虎徐小姐不捕捉老虎？ |
| bad inserts 杨大哥 | question_A_not_A | 7 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 杨大哥卖充电器不卖充电器？<br>Bad: 杨大哥卖充电器杨大哥不卖充电器？ |
| multiple edits: bad deletes 品茶; bad inserts 品茶 | question_A_not_A_daodi_a | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他到底品茶不品茶？<br>Bad: 他到底不品茶品茶？ |
| multiple edits: bad deletes 哭; bad inserts 哭 | question_A_not_A_daodi_a | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐到底哭不哭？<br>Bad: 王小姐到底不哭哭？ |
| bad inserts 吴太太 | question_A_not_A | 6 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 吴太太跨越海洋不跨越海洋？<br>Bad: 吴太太跨越海洋吴太太不跨越海洋？ |
| bad inserts 王大娘 | question_A_not_A | 6 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王大娘唱美声不唱美声？<br>Bad: 王大娘唱美声王大娘不唱美声？ |
| bad inserts 陈大姐 | question_A_not_A | 6 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 陈大姐驾驶飞机不驾驶飞机？<br>Bad: 陈大姐驾驶飞机陈大姐不驾驶飞机？ |
| multiple edits: bad deletes 入睡; bad inserts 入睡 | question_A_not_A_daodi_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张婶到底入睡不入睡？<br>Bad: 张婶到底不入睡入睡？ |
| multiple edits: bad deletes 去; bad inserts 去 | question_A_not_A_daodi_a | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 陈大姐到底去不去？<br>Bad: 陈大姐到底不去去？ |
| multiple edits: bad deletes 哭; bad inserts 哭 | question_A_not_A_daodi_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们到底哭不哭？<br>Bad: 她们到底不哭哭？ |
| multiple edits: bad deletes 游泳; bad inserts 游泳 | question_A_not_A_daodi_a | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨到底游泳不游泳？<br>Bad: 王姨到底不游泳游泳？ |
| multiple edits: bad deletes 游泳; bad inserts 游泳 | question_A_not_A_daodi_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他到底游泳不游泳？<br>Bad: 他到底不游泳游泳？ |
| multiple edits: bad deletes 看戏; bad inserts 看戏 | question_A_not_A_daodi_a | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太到底看戏不看戏？<br>Bad: 吴太太到底不看戏看戏？ |
| multiple edits: bad deletes 走; bad inserts 走 | question_A_not_A_daodi_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他到底走不走？<br>Bad: 他到底不走走？ |
| multiple edits: bad deletes 运动; bad inserts 运动 | question_A_not_A_daodi_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太到底运动不运动？<br>Bad: 吴太太到底不运动运动？ |
| bad inserts 小王 | question_A_not_A | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 小王观看动作片不观看动作片？<br>Bad: 小王观看动作片小王不观看动作片？ |
| bad inserts 李太太 | question_A_not_A | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 李太太驾驶飞机不驾驶飞机？<br>Bad: 李太太驾驶飞机李太太不驾驶飞机？ |
| bad inserts 王小姐 | question_A_not_A | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 王小姐屠宰牛不屠宰牛？<br>Bad: 王小姐屠宰牛王小姐不屠宰牛？ |
| bad inserts 胡大爷 | question_A_not_A | 5 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 胡大爷搬桌子不搬桌子？<br>Bad: 胡大爷搬桌子胡大爷不搬桌子？ |
| multiple edits: bad deletes 去; bad inserts 去 | question_A_not_A_daodi_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太到底去不去？<br>Bad: 何太太到底不去去？ |
| multiple edits: bad deletes 微笑; bad inserts 微笑 | question_A_not_A_daodi_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他到底微笑不微笑？<br>Bad: 他到底不微笑微笑？ |
| multiple edits: bad deletes 打架; bad inserts 打架 | question_A_not_A_daodi_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们到底打架不打架？<br>Bad: 她们到底不打架打架？ |
| multiple edits: bad deletes 败露; bad inserts 败露 | question_A_not_A_daodi_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你到底败露不败露？<br>Bad: 你到底不败露败露？ |
| multiple edits: bad deletes 故障; bad inserts 故障 | question_A_not_A_daodi_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥到底故障不故障？<br>Bad: 杨大哥到底不故障故障？ |
| multiple edits: bad deletes 运动; bad inserts 运动 | question_A_not_A_daodi_a | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明到底运动不运动？<br>Bad: 小明到底不运动运动？ |
| bad inserts 宋女士 | question_A_not_A | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 宋女士吃糖不吃糖？<br>Bad: 宋女士吃糖宋女士不吃糖？ |
| bad inserts 张先生 | question_A_not_A | 3 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 张先生创作小说不创作小说？<br>Bad: 张先生创作小说张先生不创作小说？ |
| multiple edits: bad deletes 笑; bad inserts 笑 | question_A_not_A_daodi_b | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你到底笑不笑？<br>Bad: 你到底不笑笑？ |

## relativization

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| bad inserts 她 | relativization_movement_no_gap | 148 | 0.8446 | 0.5878 | +0.2568 | 0.0000 | Good: 这位音乐家所鼓励的那位演员来了。<br>Bad: 这位音乐家所鼓励她的那位演员来了。 |
| multiple edits: 我 -> 谁; 。 -> ？ | relative_operator_who | 74 | 0.6486 | 0.8243 | -0.1757 | 0.0000 | Good: 张婶知晓她为什么想我打架的原因。<br>Bad: 张婶知晓她为什么想谁打架的原因？ |
| multiple edits: 她 -> 谁; 。 -> ？ | relative_operator_who | 73 | 0.8493 | 0.9589 | -0.1096 | 0.0000 | Good: 你了解小王为什么愿意她来的原因。<br>Bad: 你了解小王为什么愿意谁来的原因？ |
| multiple edits: 你 -> 谁; 。 -> ？ | relative_operator_who | 74 | 0.6216 | 0.7297 | -0.1081 | 0.0000 | Good: 张婶的朋友了解这个消费者为什么不想要你打架的原因。<br>Bad: 张婶的朋友了解这个消费者为什么不想要谁打架的原因？ |
| bad inserts 他 | relativization_movement_no_gap | 152 | 0.7039 | 0.6053 | +0.0987 | 0.0000 | Good: 陈大姐的姐妹所取代的那位吉他手来了。<br>Bad: 陈大姐的姐妹所取代他的那位吉他手来了。 |
| bad deletes 原因 | relative_operator_intepretation | 300 | 0.7967 | 0.8233 | -0.0267 | 0.0000 | Good: 她不可以把赵大爷为什么过期的原因告诉那个小孩。<br>Bad: 她不可以把赵大爷为什么过期的告诉那个小孩。 |
| multiple edits: 他 -> 谁; 。 -> ？ | relative_operator_who | 79 | 0.9494 | 0.9367 | +0.0127 | 0.0000 | Good: 我了解那位服务员不想他启程的原因。<br>Bad: 我了解那位服务员不想谁启程的原因？ |
| bad inserts 所 | relativization_movement_when_where | 300 | 0.3133 | 0.3133 | +0.0000 | 0.0000 | Good: 那个舞者呵斥你的那年，已经过去很久了。<br>Bad: 那个舞者所呵斥你的那年，已经过去很久了。 |

## topicalization

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad deletes 她在看; bad inserts 她在看 | topicalization_OSV | 3 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她在看一部教材。<br>Bad: 一部教材她在看。 |
| multiple edits: bad inserts 什么牛; bad deletes 什么牛 | topicalization_SOV_mei | 3 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你们没屠宰什么牛。<br>Bad: 你们什么牛没屠宰。 |
| multiple edits: bad deletes 你没拉; bad inserts 你没拉 | topicalization_OSV_mei | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你没拉什么小提琴。<br>Bad: 什么小提琴你没拉。 |
| multiple edits: bad deletes 她没看; bad inserts 她没看 | topicalization_OSV_mei | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她没看什么小说。<br>Bad: 什么小说她没看。 |
| multiple edits: bad inserts 什么手账; bad deletes 什么手账 | topicalization_OSV_mei | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们没看什么手账。<br>Bad: 什么手账她们没看。 |
| multiple edits: bad inserts 什么海洋; bad deletes 什么海洋 | topicalization_OSV_mei | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们没跨越什么海洋。<br>Bad: 什么海洋他们没跨越。 |
| multiple edits: bad inserts 什么火车; bad deletes 什么火车 | topicalization_OSV_mei | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们没开什么火车。<br>Bad: 什么火车她们没开。 |
| multiple edits: bad deletes 他们没拉; bad inserts 他们没拉 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们没拉什么大提琴。<br>Bad: 什么大提琴他们没拉。 |
| multiple edits: bad deletes 你们没弹; bad inserts 你们没弹 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们没弹什么玻璃珠。<br>Bad: 什么玻璃珠你们没弹。 |
| multiple edits: bad deletes 你没弹; bad inserts 你没弹 | topicalization_OSV_mei | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 你没弹什么玻璃珠。<br>Bad: 什么玻璃珠你没弹。 |
| multiple edits: bad deletes 在打断; bad inserts 在打断 | topicalization_SOV | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 它在打断一只鼻子。<br>Bad: 它一只鼻子在打断。 |
| multiple edits: bad deletes 她没拉; bad inserts 她没拉 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她没拉什么小提琴。<br>Bad: 什么小提琴她没拉。 |
| multiple edits: bad deletes 它在看; bad inserts 它在看 | topicalization_OSV | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 它在看一部教材。<br>Bad: 一部教材它在看。 |
| multiple edits: bad deletes 它没拉; bad inserts 它没拉 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 它没拉什么大提琴。<br>Bad: 什么大提琴它没拉。 |
| multiple edits: bad deletes 我们在弹; bad inserts 我们在弹 | topicalization_OSV | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我们在弹一个玻璃珠。<br>Bad: 一个玻璃珠我们在弹。 |
| multiple edits: bad deletes 我们没吃; bad inserts 我们没吃 | topicalization_OSV_mei | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我们没吃什么蛋炒饭。<br>Bad: 什么蛋炒饭我们没吃。 |
| multiple edits: bad deletes 我没盖; bad inserts 我没盖 | topicalization_OSV_mei | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 我没盖什么被子。<br>Bad: 什么被子我没盖。 |
| multiple edits: bad inserts 一个眼睛; bad deletes 一个眼睛 | topicalization_OSV | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 它在检查一个眼睛。<br>Bad: 一个眼睛它在检查。 |
| multiple edits: bad inserts 一条蛇; bad deletes 一条蛇 | topicalization_OSV | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她在捕捉一条蛇。<br>Bad: 一条蛇她在捕捉。 |
| multiple edits: bad inserts 一条被子; bad deletes 一条被子 | topicalization_OSV | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们在盖一条被子。<br>Bad: 一条被子你们在盖。 |
| multiple edits: bad inserts 什么卡车; bad deletes 什么卡车 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们没开什么卡车。<br>Bad: 什么卡车她们没开。 |
| multiple edits: bad inserts 什么头; bad deletes 什么头 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们没检查什么头。<br>Bad: 什么头他们没检查。 |
| multiple edits: bad inserts 什么歌; bad deletes 什么歌 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她没唱什么歌。<br>Bad: 什么歌她没唱。 |
| multiple edits: bad inserts 什么漫画; bad deletes 什么漫画 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们没看什么漫画。<br>Bad: 什么漫画他们没看。 |
| multiple edits: bad inserts 什么被子; bad deletes 什么被子 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们没盖什么被子。<br>Bad: 什么被子她们没盖。 |
| multiple edits: bad inserts 什么飞机; bad deletes 什么飞机 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 它没驾驶什么飞机。<br>Bad: 什么飞机它没驾驶。 |
| multiple edits: bad inserts 任何京剧; bad deletes 任何京剧 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 她们没唱任何京剧。<br>Bad: 任何京剧她们没唱。 |
| multiple edits: bad inserts 任何咖啡; bad deletes 任何咖啡 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们没喝任何咖啡。<br>Bad: 任何咖啡他们没喝。 |
| multiple edits: bad inserts 任何笛子; bad deletes 任何笛子 | topicalization_OSV_mei | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 他们没吹任何笛子。<br>Bad: 任何笛子他们没吹。 |
| multiple edits: bad inserts 一条鱼; bad deletes 一条鱼 | topicalization_OSV | 14 | 0.9286 | 0.0000 | +0.9286 | 0.0000 | Good: 他在清蒸一条鱼。<br>Bad: 一条鱼他在清蒸。 |
| multiple edits: bad deletes 在煮; bad inserts 在煮 | topicalization_SOV | 4 | 0.2500 | 1.0000 | -0.7500 | 0.0000 | Good: 它在煮一只鸭。<br>Bad: 它一只鸭在煮。 |
| multiple edits: bad inserts 什么轮船; bad deletes 什么轮船 | topicalization_OSV_mei | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 她们没驾驶什么轮船。<br>Bad: 什么轮船她们没驾驶。 |
| multiple edits: bad deletes 在烧; bad inserts 在烧 | topicalization_SOV | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 他在烧一只鸭。<br>Bad: 他一只鸭在烧。 |
| multiple edits: bad inserts 一只手; bad deletes 一只手 | topicalization_OSV | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 它在包扎一只手。<br>Bad: 一只手它在包扎。 |
| multiple edits: bad inserts 什么耳朵; bad deletes 什么耳朵 | topicalization_OSV_mei | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 它没检查什么耳朵。<br>Bad: 什么耳朵它没检查。 |
| multiple edits: bad inserts 什么货车; bad deletes 什么货车 | topicalization_OSV_mei | 3 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 她们没开什么货车。<br>Bad: 什么货车她们没开。 |
| multiple edits: bad deletes 在麻醉; bad inserts 在麻醉 | topicalization_SOV | 9 | 0.0000 | 0.6667 | -0.6667 | 0.0000 | Good: 他们在麻醉一只老虎。<br>Bad: 他们一只老虎在麻醉。 |
| multiple edits: bad deletes 他没开; bad inserts 他没开 | topicalization_OSV_mei | 3 | 0.6667 | 0.0000 | +0.6667 | 0.0000 | Good: 他没开什么轮船。<br>Bad: 什么轮船他没开。 |
| multiple edits: bad inserts 一片面包; bad deletes 一片面包 | topicalization_OSV | 3 | 0.6667 | 0.0000 | +0.6667 | 0.0000 | Good: 你们在吃一片面包。<br>Bad: 一片面包你们在吃。 |
| multiple edits: bad deletes 在捕捉; bad inserts 在捕捉 | topicalization_SOV | 8 | 0.3750 | 1.0000 | -0.6250 | 0.0000 | Good: 我们在捕捉一头大象。<br>Bad: 我们一头大象在捕捉。 |
| multiple edits: bad deletes 没弹; bad inserts 没弹 | topicalization_SOV_mei | 8 | 0.8750 | 0.2500 | +0.6250 | 0.0000 | Good: 他们没弹任何玻璃珠。<br>Bad: 他们任何玻璃珠没弹。 |
| multiple edits: bad inserts 一本教材; bad deletes 一本教材 | topicalization_OSV | 8 | 0.6250 | 0.0000 | +0.6250 | 0.0000 | Good: 它在预习一本教材。<br>Bad: 一本教材它在预习。 |
| multiple edits: bad deletes 在弹; bad inserts 在弹 | topicalization_SOV | 5 | 0.6000 | 0.0000 | +0.6000 | 0.0000 | Good: 你们在弹一个玻璃珠。<br>Bad: 你们一个玻璃珠在弹。 |
| multiple edits: bad inserts 任何教材; bad deletes 任何教材 | topicalization_OSV_mei | 5 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 她没预习任何教材。<br>Bad: 任何教材她没预习。 |
| multiple edits: bad inserts 一条鱼; bad deletes 一条鱼 | topicalization_SOV | 10 | 0.3000 | 0.8000 | -0.5000 | 0.0000 | Good: 她们在清蒸一条鱼。<br>Bad: 她们一条鱼在清蒸。 |
| multiple edits: bad deletes 他没唱; bad inserts 他没唱 | topicalization_OSV_mei | 4 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他没唱什么戏曲。<br>Bad: 什么戏曲他没唱。 |
| multiple edits: bad deletes 他没拉; bad inserts 他没拉 | topicalization_OSV_mei | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他没拉什么小提琴。<br>Bad: 什么小提琴他没拉。 |
| multiple edits: bad deletes 他没盖; bad inserts 他没盖 | topicalization_OSV_mei | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 他没盖什么被子。<br>Bad: 什么被子他没盖。 |
| multiple edits: bad deletes 它没开; bad inserts 它没开 | topicalization_OSV_mei | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 它没开任何火车。<br>Bad: 任何火车它没开。 |
| multiple edits: bad deletes 它没盖; bad inserts 它没盖 | topicalization_OSV_mei | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 它没盖什么被子。<br>Bad: 什么被子它没盖。 |
| multiple edits: bad deletes 我没看; bad inserts 我没看 | topicalization_OSV_mei | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 我没看什么录像带。<br>Bad: 什么录像带我没看。 |
| multiple edits: bad inserts 一只手; bad deletes 一只手 | topicalization_SOV | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 他在打断一只手。<br>Bad: 他一只手在打断。 |
| multiple edits: bad inserts 一只老虎; bad deletes 一只老虎 | topicalization_OSV | 2 | 0.0000 | 0.5000 | -0.5000 | 0.0000 | Good: 她们在麻醉一只老虎。<br>Bad: 一只老虎她们在麻醉。 |
| multiple edits: bad inserts 什么头; bad deletes 什么头 | topicalization_SOV_mei | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 我没包扎什么头。<br>Bad: 我什么头没包扎。 |
| multiple edits: bad inserts 什么小狗; bad deletes 什么小狗 | topicalization_OSV_mei | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他没领养什么小狗。<br>Bad: 什么小狗他没领养。 |
| multiple edits: bad inserts 什么沙漠; bad deletes 什么沙漠 | topicalization_OSV_mei | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他没跨越什么沙漠。<br>Bad: 什么沙漠他没跨越。 |
| multiple edits: bad inserts 任何火车; bad deletes 任何火车 | topicalization_OSV_mei | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 它没驾驶任何火车。<br>Bad: 任何火车它没驾驶。 |
| multiple edits: bad inserts 任何货车; bad deletes 任何货车 | topicalization_OSV_mei | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你没驾驶任何货车。<br>Bad: 任何货车你没驾驶。 |
| multiple edits: bad deletes 没煮; bad inserts 没煮 | topicalization_SOV_mei | 9 | 1.0000 | 0.5556 | +0.4444 | 0.0000 | Good: 我们没煮什么鸡。<br>Bad: 我们什么鸡没煮。 |
| multiple edits: bad inserts 一头牛; bad deletes 一头牛 | topicalization_OSV | 7 | 0.5714 | 1.0000 | -0.4286 | 0.0000 | Good: 他在屠宰一头牛。<br>Bad: 一头牛他在屠宰。 |
| multiple edits: bad inserts 什么鱼; bad deletes 什么鱼 | topicalization_OSV_mei | 10 | 0.8000 | 0.4000 | +0.4000 | 0.0000 | Good: 她没烧什么鱼。<br>Bad: 什么鱼她没烧。 |
| multiple edits: bad inserts 什么杯子; bad deletes 什么杯子 | topicalization_OSV_mei | 5 | 0.8000 | 0.4000 | +0.4000 | 0.0000 | Good: 她没清洗什么杯子。<br>Bad: 什么杯子她没清洗。 |
| multiple edits: bad deletes 在炖; bad inserts 在炖 | topicalization_SOV | 5 | 0.2000 | 0.6000 | -0.4000 | 0.0000 | Good: 他们在炖一只鸭。<br>Bad: 他们一只鸭在炖。 |
| multiple edits: bad inserts 一头大象; bad deletes 一头大象 | topicalization_OSV | 13 | 0.4615 | 0.8462 | -0.3846 | 0.0000 | Good: 它在麻醉一头大象。<br>Bad: 一头大象它在麻醉。 |
| multiple edits: bad inserts 一只鸭; bad deletes 一只鸭 | topicalization_OSV | 11 | 0.7273 | 0.3636 | +0.3636 | 0.0000 | Good: 它在爆炒一只鸭。<br>Bad: 一只鸭它在爆炒。 |
| multiple edits: bad deletes 它没喝; bad inserts 它没喝 | topicalization_OSV_mei | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 它没喝任何啤酒。<br>Bad: 任何啤酒它没喝。 |
| multiple edits: bad inserts 一块糖果; bad deletes 一块糖果 | topicalization_OSV | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 我们在吃一块糖果。<br>Bad: 一块糖果我们在吃。 |
| multiple edits: bad inserts 一块蛋糕; bad deletes 一块蛋糕 | topicalization_OSV | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 我们在吃一块蛋糕。<br>Bad: 一块蛋糕我们在吃。 |
| multiple edits: bad inserts 什么教材; bad deletes 什么教材 | topicalization_OSV_mei | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 他们没预习什么教材。<br>Bad: 什么教材他们没预习。 |
| multiple edits: bad inserts 什么脚; bad deletes 什么脚 | topicalization_OSV_mei | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 你没打断什么脚。<br>Bad: 什么脚你没打断。 |
| multiple edits: bad deletes 在包扎; bad inserts 在包扎 | topicalization_SOV | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 你在包扎一个耳朵。<br>Bad: 你一个耳朵在包扎。 |
| multiple edits: bad inserts 一只脚; bad deletes 一只脚 | topicalization_OSV | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 她在包扎一只脚。<br>Bad: 一只脚她在包扎。 |
| multiple edits: bad inserts 一只脚; bad deletes 一只脚 | topicalization_SOV | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 你们在检查一只脚。<br>Bad: 你们一只脚在检查。 |
| multiple edits: bad inserts 一只鸡; bad deletes 一只鸡 | topicalization_SOV | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 我们在爆炒一只鸡。<br>Bad: 我们一只鸡在爆炒。 |
| multiple edits: bad inserts 一本小说; bad deletes 一本小说 | topicalization_OSV | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 我在创作一本小说。<br>Bad: 一本小说我在创作。 |
| multiple edits: bad inserts 一条腿; bad deletes 一条腿 | topicalization_SOV | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 你们在打断一条腿。<br>Bad: 你们一条腿在打断。 |
| multiple edits: bad inserts 什么大象; bad deletes 什么大象 | topicalization_OSV_mei | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 我们没麻醉什么大象。<br>Bad: 什么大象我们没麻醉。 |
| multiple edits: bad inserts 什么牛; bad deletes 什么牛 | topicalization_OSV_mei | 3 | 0.3333 | 0.6667 | -0.3333 | 0.0000 | Good: 我们没屠宰什么牛。<br>Bad: 什么牛我们没屠宰。 |
| multiple edits: bad deletes 在观看; bad inserts 在观看 | topicalization_SOV | 7 | 0.1429 | 0.4286 | -0.2857 | 0.0000 | Good: 我在观看一部电影。<br>Bad: 我一部电影在观看。 |
| multiple edits: bad deletes 在预习; bad inserts 在预习 | topicalization_SOV | 7 | 0.2857 | 0.0000 | +0.2857 | 0.0000 | Good: 她们在预习一本教材。<br>Bad: 她们一本教材在预习。 |
| multiple edits: bad inserts 什么鸡; bad deletes 什么鸡 | topicalization_OSV_mei | 7 | 0.5714 | 0.2857 | +0.2857 | 0.0000 | Good: 他没煮什么鸡。<br>Bad: 什么鸡他没煮。 |
| multiple edits: bad inserts 任何鸭; bad deletes 任何鸭 | topicalization_OSV_mei | 12 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 你没炖任何鸭。<br>Bad: 任何鸭你没炖。 |
| multiple edits: bad deletes 没炖; bad inserts 没炖 | topicalization_SOV_mei | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 你们没炖什么鸡。<br>Bad: 你们什么鸡没炖。 |
| multiple edits: bad deletes 没预习; bad inserts 没预习 | topicalization_SOV_mei | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 它没预习什么教材。<br>Bad: 它什么教材没预习。 |
| multiple edits: bad deletes 在检查; bad inserts 在检查 | topicalization_SOV | 4 | 0.2500 | 0.5000 | -0.2500 | 0.0000 | Good: 他在检查一只鼻子。<br>Bad: 他一只鼻子在检查。 |
| multiple edits: bad deletes 她在吃; bad inserts 她在吃 | topicalization_OSV | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 她在吃一片面包。<br>Bad: 一片面包她在吃。 |
| multiple edits: bad deletes 没捕捉; bad inserts 没捕捉 | topicalization_SOV_mei | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 他没捕捉什么老虎。<br>Bad: 他什么老虎没捕捉。 |
| multiple edits: bad inserts 一只鸭; bad deletes 一只鸭 | topicalization_SOV | 4 | 0.2500 | 0.5000 | -0.2500 | 0.0000 | Good: 我们在清蒸一只鸭。<br>Bad: 我们一只鸭在清蒸。 |
| multiple edits: bad inserts 什么脚; bad deletes 什么脚 | topicalization_SOV_mei | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 你没打断什么脚。<br>Bad: 你什么脚没打断。 |
| multiple edits: bad inserts 什么鸭; bad deletes 什么鸭 | topicalization_SOV_mei | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 你没清蒸什么鸭。<br>Bad: 你什么鸭没清蒸。 |
| multiple edits: bad inserts 什么鼻子; bad deletes 什么鼻子 | topicalization_OSV_mei | 4 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 她没打断什么鼻子。<br>Bad: 什么鼻子她没打断。 |
| multiple edits: bad inserts 一条腿; bad deletes 一条腿 | topicalization_OSV | 9 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 他在包扎一条腿。<br>Bad: 一条腿他在包扎。 |
| multiple edits: bad inserts 什么鸡; bad deletes 什么鸡 | topicalization_SOV_mei | 10 | 0.6000 | 0.8000 | -0.2000 | 0.0000 | Good: 她没清蒸什么鸡。<br>Bad: 她什么鸡没清蒸。 |
| multiple edits: bad deletes 没包扎; bad inserts 没包扎 | topicalization_SOV_mei | 5 | 0.8000 | 0.6000 | +0.2000 | 0.0000 | Good: 你们没包扎什么耳朵。<br>Bad: 你们什么耳朵没包扎。 |
| multiple edits: bad inserts 任何老虎; bad deletes 任何老虎 | topicalization_OSV_mei | 5 | 0.8000 | 0.6000 | +0.2000 | 0.0000 | Good: 她们没麻醉任何老虎。<br>Bad: 任何老虎她们没麻醉。 |
| multiple edits: bad deletes 在创作; bad inserts 在创作 | topicalization_SOV | 15 | 0.5333 | 0.7333 | -0.2000 | 0.0000 | Good: 他们在创作一本漫画。<br>Bad: 他们一本漫画在创作。 |
| multiple edits: bad inserts 什么鱼; bad deletes 什么鱼 | topicalization_SOV_mei | 6 | 0.8333 | 0.6667 | +0.1667 | 0.0000 | Good: 你们没清蒸什么鱼。<br>Bad: 你们什么鱼没清蒸。 |
| multiple edits: bad inserts 一头牛; bad deletes 一头牛 | topicalization_SOV | 6 | 0.3333 | 0.5000 | -0.1667 | 0.0000 | Good: 我们在屠宰一头牛。<br>Bad: 我们一头牛在屠宰。 |
| multiple edits: bad inserts 任何鸡; bad deletes 任何鸡 | topicalization_OSV_mei | 12 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 它没炖任何鸡。<br>Bad: 任何鸡它没炖。 |
| multiple edits: bad deletes 他在吃; bad inserts 他在吃 | topicalization_OSV | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 他在吃一片面包。<br>Bad: 一片面包他在吃。 |
| multiple edits: bad deletes 没领养; bad inserts 没领养 | topicalization_SOV_mei | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 她们没领养什么小狗。<br>Bad: 她们什么小狗没领养。 |
| multiple edits: bad inserts 一本漫画; bad deletes 一本漫画 | topicalization_OSV | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 它在创作一本漫画。<br>Bad: 一本漫画它在创作。 |
| multiple edits: bad inserts 一本手账; bad deletes 一本手账 | topicalization_OSV | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 我们在看一本手账。<br>Bad: 一本手账我们在看。 |
| multiple edits: bad inserts 一部小说; bad deletes 一部小说 | topicalization_OSV | 7 | 0.8571 | 1.0000 | -0.1429 | 0.0000 | Good: 我们在创作一部小说。<br>Bad: 一部小说我们在创作。 |
| multiple edits: bad deletes 在制作; bad inserts 在制作 | topicalization_SOV | 7 | 0.5714 | 0.4286 | +0.1429 | 0.0000 | Good: 他在制作一部电影。<br>Bad: 他一部电影在制作。 |
| multiple edits: bad deletes 在拍摄; bad inserts 在拍摄 | topicalization_SOV | 7 | 0.7143 | 0.8571 | -0.1429 | 0.0000 | Good: 我在拍摄一部电影。<br>Bad: 我一部电影在拍摄。 |
| multiple edits: bad deletes 在吃; bad inserts 在吃 | topicalization_SOV | 66 | 0.4697 | 0.6061 | -0.1364 | 0.0000 | Good: 它在吃一桶方便面。<br>Bad: 它一桶方便面在吃。 |
| multiple edits: bad deletes 它在吃; bad inserts 它在吃 | topicalization_OSV | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 它在吃一片面包。<br>Bad: 一片面包它在吃。 |
| multiple edits: bad deletes 没看; bad inserts 没看 | topicalization_SOV_mei | 9 | 1.0000 | 0.8889 | +0.1111 | 0.0000 | Good: 他们没看什么教材。<br>Bad: 他们什么教材没看。 |
| multiple edits: bad inserts 一个头; bad deletes 一个头 | topicalization_SOV | 9 | 0.1111 | 0.2222 | -0.1111 | 0.0000 | Good: 我在包扎一个头。<br>Bad: 我一个头在包扎。 |
| multiple edits: bad deletes 在领养; bad inserts 在领养 | topicalization_SOV | 9 | 0.5556 | 0.6667 | -0.1111 | 0.0000 | Good: 他们在领养一条小狗。<br>Bad: 他们一条小狗在领养。 |
| multiple edits: bad inserts 什么鸭; bad deletes 什么鸭 | topicalization_OSV_mei | 10 | 0.7000 | 0.8000 | -0.1000 | 0.0000 | Good: 我没清蒸什么鸭。<br>Bad: 什么鸭我没清蒸。 |
| multiple edits: bad deletes 没拉; bad inserts 没拉 | topicalization_SOV_mei | 10 | 1.0000 | 0.9000 | +0.1000 | 0.0000 | Good: 我们没拉任何小提琴。<br>Bad: 我们任何小提琴没拉。 |
| multiple edits: bad deletes 没跨越; bad inserts 没跨越 | topicalization_SOV_mei | 11 | 1.0000 | 0.9091 | +0.0909 | 0.0000 | Good: 你没跨越任何海洋。<br>Bad: 你任何海洋没跨越。 |
| multiple edits: bad deletes 在喝; bad inserts 在喝 | topicalization_SOV | 78 | 0.6538 | 0.7436 | -0.0897 | 0.0000 | Good: 他在喝一杯红酒。<br>Bad: 他一杯红酒在喝。 |
| multiple edits: bad deletes 在看; bad inserts 在看 | topicalization_SOV | 14 | 0.5714 | 0.6429 | -0.0714 | 0.0000 | Good: 你在看一本手账。<br>Bad: 你一本手账在看。 |
| multiple edits: bad inserts 任何鱼; bad deletes 任何鱼 | topicalization_OSV_mei | 14 | 0.9286 | 0.8571 | +0.0714 | 0.0000 | Good: 我没烧任何鱼。<br>Bad: 任何鱼我没烧。 |
| multiple edits: bad deletes 没吃; bad inserts 没吃 | topicalization_SOV_mei | 14 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没吃任何蛋炒饭。<br>Bad: 他们任何蛋炒饭没吃。 |
| multiple edits: bad deletes 没唱; bad inserts 没唱 | topicalization_SOV_mei | 14 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没唱任何京剧。<br>Bad: 她们任何京剧没唱。 |
| multiple edits: bad deletes 没开; bad inserts 没开 | topicalization_SOV_mei | 14 | 0.9286 | 0.9286 | +0.0000 | 0.0000 | Good: 我们没开任何卡车。<br>Bad: 我们任何卡车没开。 |
| multiple edits: bad deletes 没观看; bad inserts 没观看 | topicalization_SOV_mei | 13 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没观看什么电影。<br>Bad: 他什么电影没观看。 |
| multiple edits: bad inserts 一只鸡; bad deletes 一只鸡 | topicalization_OSV | 13 | 0.2308 | 0.2308 | +0.0000 | 0.0000 | Good: 我们在煮一只鸡。<br>Bad: 一只鸡我们在煮。 |
| multiple edits: bad deletes 没喝; bad inserts 没喝 | topicalization_SOV_mei | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没喝什么红酒。<br>Bad: 我什么红酒没喝。 |
| multiple edits: bad deletes 没烧; bad inserts 没烧 | topicalization_SOV_mei | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没烧任何鱼。<br>Bad: 他任何鱼没烧。 |
| multiple edits: bad deletes 没驾驶; bad inserts 没驾驶 | topicalization_SOV_mei | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没驾驶任何飞机。<br>Bad: 它任何飞机没驾驶。 |
| multiple edits: bad inserts 一部电影; bad deletes 一部电影 | topicalization_OSV | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在拍摄一部电影。<br>Bad: 一部电影你们在拍摄。 |
| multiple edits: bad deletes 没创作; bad inserts 没创作 | topicalization_SOV_mei | 11 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没创作任何漫画。<br>Bad: 你任何漫画没创作。 |
| multiple edits: bad deletes 它在喝; bad inserts 它在喝 | topicalization_OSV | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它在喝一杯咖啡。<br>Bad: 一杯咖啡它在喝。 |
| multiple edits: bad deletes 没演奏; bad inserts 没演奏 | topicalization_SOV_mei | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没演奏什么华尔兹。<br>Bad: 他们什么华尔兹没演奏。 |
| multiple edits: bad deletes 没盖; bad inserts 没盖 | topicalization_SOV_mei | 10 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没盖什么被子。<br>Bad: 你们什么被子没盖。 |
| multiple edits: bad deletes 他在喝; bad inserts 他在喝 | topicalization_OSV | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他在喝一瓶冰红茶。<br>Bad: 一瓶冰红茶他在喝。 |
| multiple edits: bad deletes 你在喝; bad inserts 你在喝 | topicalization_OSV | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你在喝一瓶冰红茶。<br>Bad: 一瓶冰红茶你在喝。 |
| multiple edits: bad deletes 没拍摄; bad inserts 没拍摄 | topicalization_SOV_mei | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没拍摄什么电影。<br>Bad: 她们什么电影没拍摄。 |
| multiple edits: bad deletes 没麻醉; bad inserts 没麻醉 | topicalization_SOV_mei | 8 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 你们没麻醉什么大象。<br>Bad: 你们什么大象没麻醉。 |
| multiple edits: bad inserts 任何电影; bad deletes 任何电影 | topicalization_OSV_mei | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没拍摄任何电影。<br>Bad: 任何电影他们没拍摄。 |
| multiple edits: bad inserts 什么电影; bad deletes 什么电影 | topicalization_OSV_mei | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没观看什么电影。<br>Bad: 什么电影她没观看。 |
| multiple edits: bad deletes 你在吃; bad inserts 你在吃 | topicalization_OSV | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你在吃一串香蕉。<br>Bad: 一串香蕉你在吃。 |
| multiple edits: bad deletes 我在喝; bad inserts 我在喝 | topicalization_OSV | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我在喝一瓶啤酒。<br>Bad: 一瓶啤酒我在喝。 |
| multiple edits: bad deletes 没制作; bad inserts 没制作 | topicalization_SOV_mei | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没制作任何视频。<br>Bad: 他们任何视频没制作。 |
| multiple edits: bad inserts 一个头; bad deletes 一个头 | topicalization_OSV | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在检查一个头。<br>Bad: 一个头我们在检查。 |
| multiple edits: bad inserts 一个耳朵; bad deletes 一个耳朵 | topicalization_OSV | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他在包扎一个耳朵。<br>Bad: 一个耳朵他在包扎。 |
| multiple edits: bad inserts 任何鸭; bad deletes 任何鸭 | topicalization_SOV_mei | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没清蒸任何鸭。<br>Bad: 我任何鸭没清蒸。 |
| multiple edits: bad deletes 她在喝; bad inserts 她在喝 | topicalization_OSV | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她在喝一瓶白酒。<br>Bad: 一瓶白酒她在喝。 |
| multiple edits: bad deletes 没吹; bad inserts 没吹 | topicalization_SOV_mei | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没吹什么双簧。<br>Bad: 她什么双簧没吹。 |
| multiple edits: bad inserts 一个杯子; bad deletes 一个杯子 | topicalization_OSV | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在清洗一个杯子。<br>Bad: 一个杯子我们在清洗。 |
| multiple edits: bad inserts 一块糖; bad deletes 一块糖 | topicalization_OSV | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在吃一块糖。<br>Bad: 一块糖你们在吃。 |
| multiple edits: bad inserts 一部漫画; bad deletes 一部漫画 | topicalization_OSV | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我在创作一部漫画。<br>Bad: 一部漫画我在创作。 |
| multiple edits: bad inserts 任何漫画; bad deletes 任何漫画 | topicalization_OSV_mei | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没创作任何漫画。<br>Bad: 任何漫画她没创作。 |
| multiple edits: bad inserts 任何牛; bad deletes 任何牛 | topicalization_OSV_mei | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没屠宰任何牛。<br>Bad: 任何牛你没屠宰。 |
| multiple edits: bad inserts 任何牛; bad deletes 任何牛 | topicalization_SOV_mei | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没屠宰任何牛。<br>Bad: 她任何牛没屠宰。 |
| multiple edits: bad deletes 在清洗; bad inserts 在清洗 | topicalization_SOV | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在清洗一个杯子。<br>Bad: 我们一个杯子在清洗。 |
| multiple edits: bad deletes 我在吃; bad inserts 我在吃 | topicalization_OSV | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我在吃一串香蕉。<br>Bad: 一串香蕉我在吃。 |
| multiple edits: bad deletes 没清洗; bad inserts 没清洗 | topicalization_SOV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没清洗任何杯子。<br>Bad: 我任何杯子没清洗。 |
| multiple edits: bad inserts 一串香蕉; bad deletes 一串香蕉 | topicalization_OSV | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在吃一串香蕉。<br>Bad: 一串香蕉我们在吃。 |
| multiple edits: bad inserts 任何头; bad deletes 任何头 | topicalization_SOV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没检查任何头。<br>Bad: 你任何头没检查。 |
| multiple edits: bad inserts 任何小狗; bad deletes 任何小狗 | topicalization_OSV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没领养任何小狗。<br>Bad: 任何小狗我们没领养。 |
| multiple edits: bad inserts 任何杯子; bad deletes 任何杯子 | topicalization_OSV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没清洗任何杯子。<br>Bad: 任何杯子他们没清洗。 |
| multiple edits: bad inserts 任何沙漠; bad deletes 任何沙漠 | topicalization_OSV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没跨越任何沙漠。<br>Bad: 任何沙漠你们没跨越。 |
| multiple edits: bad inserts 任何海洋; bad deletes 任何海洋 | topicalization_OSV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没跨越任何海洋。<br>Bad: 任何海洋你没跨越。 |
| multiple edits: bad inserts 任何腿; bad deletes 任何腿 | topicalization_SOV_mei | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没包扎任何腿。<br>Bad: 我任何腿没包扎。 |
| multiple edits: bad deletes 他们在喝; bad inserts 他们在喝 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们在喝一杯葡萄汁。<br>Bad: 一杯葡萄汁他们在喝。 |
| multiple edits: bad deletes 在盖; bad inserts 在盖 | topicalization_SOV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你在盖一条被子。<br>Bad: 你一条被子在盖。 |
| multiple edits: bad deletes 她们在喝; bad inserts 她们在喝 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在喝一瓶矿泉水。<br>Bad: 一瓶矿泉水她们在喝。 |
| multiple edits: bad deletes 没检查; bad inserts 没检查 | topicalization_SOV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没检查任何眼睛。<br>Bad: 我们任何眼睛没检查。 |
| multiple edits: bad inserts 一只小猫; bad deletes 一只小猫 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他在领养一只小猫。<br>Bad: 一只小猫他在领养。 |
| multiple edits: bad inserts 一只鼻子; bad deletes 一只鼻子 | topicalization_OSV | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 它在打断一只鼻子。<br>Bad: 一只鼻子它在打断。 |
| multiple edits: bad inserts 一条小狗; bad deletes 一条小狗 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在领养一条小狗。<br>Bad: 一条小狗你们在领养。 |
| multiple edits: bad inserts 一杯咖啡; bad deletes 一杯咖啡 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在喝一杯咖啡。<br>Bad: 一杯咖啡你们在喝。 |
| multiple edits: bad inserts 一杯橙汁; bad deletes 一杯橙汁 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在喝一杯橙汁。<br>Bad: 一杯橙汁我们在喝。 |
| multiple edits: bad inserts 一桶啤酒; bad deletes 一桶啤酒 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在喝一桶啤酒。<br>Bad: 一桶啤酒你们在喝。 |
| multiple edits: bad inserts 一瓶白酒; bad deletes 一瓶白酒 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在喝一瓶白酒。<br>Bad: 一瓶白酒她们在喝。 |
| multiple edits: bad inserts 一瓶红酒; bad deletes 一瓶红酒 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在喝一瓶红酒。<br>Bad: 一瓶红酒你们在喝。 |
| multiple edits: bad inserts 一部日记; bad deletes 一部日记 | topicalization_OSV | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在看一部日记。<br>Bad: 一部日记你们在看。 |
| multiple edits: bad inserts 什么双簧; bad deletes 什么双簧 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没吹什么双簧。<br>Bad: 什么双簧她们没吹。 |
| multiple edits: bad inserts 什么手; bad deletes 什么手 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没包扎什么手。<br>Bad: 什么手他们没包扎。 |
| multiple edits: bad inserts 什么老虎; bad deletes 什么老虎 | topicalization_OSV_mei | 3 | 0.3333 | 0.3333 | +0.0000 | 0.0000 | Good: 你们没捕捉什么老虎。<br>Bad: 什么老虎你们没捕捉。 |
| multiple edits: bad inserts 什么腿; bad deletes 什么腿 | topicalization_SOV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没打断什么腿。<br>Bad: 他什么腿没打断。 |
| multiple edits: bad inserts 任何书; bad deletes 任何书 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没看任何书。<br>Bad: 任何书他没看。 |
| multiple edits: bad inserts 任何小猫; bad deletes 任何小猫 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没领养任何小猫。<br>Bad: 任何小猫他们没领养。 |
| multiple edits: bad inserts 任何小说; bad deletes 任何小说 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没创作任何小说。<br>Bad: 任何小说它没创作。 |
| multiple edits: bad inserts 任何歌曲; bad deletes 任何歌曲 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没演奏任何歌曲。<br>Bad: 任何歌曲他们没演奏。 |
| multiple edits: bad inserts 任何鱼; bad deletes 任何鱼 | topicalization_SOV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没清蒸任何鱼。<br>Bad: 她们任何鱼没清蒸。 |
| multiple edits: bad inserts 任何鸡; bad deletes 任何鸡 | topicalization_SOV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没爆炒任何鸡。<br>Bad: 他们任何鸡没爆炒。 |
| multiple edits: bad inserts 任何鼻子; bad deletes 任何鼻子 | topicalization_OSV_mei | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没打断任何鼻子。<br>Bad: 任何鼻子她没打断。 |
| multiple edits: bad deletes 他没演奏; bad inserts 他没演奏 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没演奏任何狂想曲。<br>Bad: 任何狂想曲他没演奏。 |
| multiple edits: bad deletes 你们在喝; bad inserts 你们在喝 | topicalization_OSV | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在喝一瓶冰红茶。<br>Bad: 一瓶冰红茶你们在喝。 |
| multiple edits: bad deletes 你没盖; bad inserts 你没盖 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没盖任何被子。<br>Bad: 任何被子你没盖。 |
| multiple edits: bad deletes 它没吃; bad inserts 它没吃 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没吃任何糖果。<br>Bad: 任何糖果它没吃。 |
| multiple edits: bad deletes 没打断; bad inserts 没打断 | topicalization_SOV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没打断任何鼻子。<br>Bad: 你任何鼻子没打断。 |
| multiple edits: bad inserts 一本书; bad deletes 一本书 | topicalization_OSV | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在看一本书。<br>Bad: 一本书她们在看。 |
| multiple edits: bad inserts 一杯白酒; bad deletes 一杯白酒 | topicalization_OSV | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在喝一杯白酒。<br>Bad: 一杯白酒她们在喝。 |
| multiple edits: bad inserts 一瓶可乐; bad deletes 一瓶可乐 | topicalization_OSV | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在喝一瓶可乐。<br>Bad: 一瓶可乐她们在喝。 |
| multiple edits: bad inserts 一瓶啤酒; bad deletes 一瓶啤酒 | topicalization_OSV | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们在喝一瓶啤酒。<br>Bad: 一瓶啤酒他们在喝。 |
| multiple edits: bad inserts 一部视频; bad deletes 一部视频 | topicalization_OSV | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在制作一部视频。<br>Bad: 一部视频她们在制作。 |
| multiple edits: bad inserts 什么书; bad deletes 什么书 | topicalization_OSV_mei | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 她没看什么书。<br>Bad: 什么书她没看。 |
| multiple edits: bad inserts 什么小说; bad deletes 什么小说 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没看什么小说。<br>Bad: 什么小说你们没看。 |
| multiple edits: bad inserts 什么歌曲; bad deletes 什么歌曲 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没演奏什么歌曲。<br>Bad: 什么歌曲你没演奏。 |
| multiple edits: bad inserts 什么胃; bad deletes 什么胃 | topicalization_SOV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没检查什么胃。<br>Bad: 你们什么胃没检查。 |
| multiple edits: bad inserts 任何双簧; bad deletes 任何双簧 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没吹任何双簧。<br>Bad: 任何双簧他们没吹。 |
| multiple edits: bad inserts 任何大象; bad deletes 任何大象 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没麻醉任何大象。<br>Bad: 任何大象你们没麻醉。 |
| multiple edits: bad inserts 任何手; bad deletes 任何手 | topicalization_SOV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没检查任何手。<br>Bad: 她任何手没检查。 |
| multiple edits: bad inserts 任何手账; bad deletes 任何手账 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没制作任何手账。<br>Bad: 任何手账她们没制作。 |
| multiple edits: bad inserts 任何红酒; bad deletes 任何红酒 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没喝任何红酒。<br>Bad: 任何红酒我们没喝。 |
| multiple edits: bad inserts 任何脚; bad deletes 任何脚 | topicalization_OSV_mei | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没打断任何脚。<br>Bad: 任何脚你没打断。 |
| multiple edits: bad deletes 他们在吃; bad inserts 他们在吃 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们在吃一桶方便面。<br>Bad: 一桶方便面他们在吃。 |
| multiple edits: bad deletes 他们没喝; bad inserts 他们没喝 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没喝什么矿泉水。<br>Bad: 什么矿泉水他们没喝。 |
| multiple edits: bad deletes 他在弹; bad inserts 他在弹 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他在弹一个玻璃珠。<br>Bad: 一个玻璃珠他在弹。 |
| multiple edits: bad deletes 你们在吃; bad inserts 你们在吃 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在吃一桶方便面。<br>Bad: 一桶方便面你们在吃。 |
| multiple edits: bad deletes 你们在弹; bad inserts 你们在弹 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在弹一个玻璃珠。<br>Bad: 一个玻璃珠你们在弹。 |
| multiple edits: bad deletes 你们没吃; bad inserts 你们没吃 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没吃什么方便面。<br>Bad: 什么方便面你们没吃。 |
| multiple edits: bad deletes 你们没喝; bad inserts 你们没喝 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没喝任何冰红茶。<br>Bad: 任何冰红茶你们没喝。 |
| multiple edits: bad deletes 你们没看; bad inserts 你们没看 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没看什么录像带。<br>Bad: 什么录像带你们没看。 |
| multiple edits: bad deletes 你在弹; bad inserts 你在弹 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你在弹一个玻璃珠。<br>Bad: 一个玻璃珠你在弹。 |
| multiple edits: bad deletes 你没唱; bad inserts 你没唱 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没唱任何戏曲。<br>Bad: 任何戏曲你没唱。 |
| multiple edits: bad deletes 你没喝; bad inserts 你没喝 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没喝任何咖啡。<br>Bad: 任何咖啡你没喝。 |
| multiple edits: bad deletes 你没演奏; bad inserts 你没演奏 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没演奏任何狂想曲。<br>Bad: 任何狂想曲你没演奏。 |
| multiple edits: bad deletes 她们在吃; bad inserts 她们在吃 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在吃一桶方便面。<br>Bad: 一桶方便面她们在吃。 |
| multiple edits: bad deletes 她们在弹; bad inserts 她们在弹 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在弹一个玻璃珠。<br>Bad: 一个玻璃珠她们在弹。 |
| multiple edits: bad deletes 她在弹; bad inserts 她在弹 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她在弹一个玻璃珠。<br>Bad: 一个玻璃珠她在弹。 |
| multiple edits: bad deletes 她在拍摄; bad inserts 她在拍摄 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她在拍摄一部动作片。<br>Bad: 一部动作片她在拍摄。 |
| multiple edits: bad deletes 她在观看; bad inserts 她在观看 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她在观看一部动作片。<br>Bad: 一部动作片她在观看。 |
| multiple edits: bad deletes 她没吹; bad inserts 她没吹 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没吹什么双簧。<br>Bad: 什么双簧她没吹。 |
| multiple edits: bad deletes 她没唱; bad inserts 她没唱 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没唱什么美声。<br>Bad: 什么美声她没唱。 |
| multiple edits: bad deletes 她没喝; bad inserts 她没喝 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没喝任何红酒。<br>Bad: 任何红酒她没喝。 |
| multiple edits: bad deletes 她没演奏; bad inserts 她没演奏 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没演奏任何奏鸣曲。<br>Bad: 任何奏鸣曲她没演奏。 |
| multiple edits: bad deletes 她没观看; bad inserts 她没观看 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没观看任何记录片。<br>Bad: 任何记录片她没观看。 |
| multiple edits: bad deletes 它在弹; bad inserts 它在弹 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它在弹一个玻璃珠。<br>Bad: 一个玻璃珠它在弹。 |
| multiple edits: bad deletes 它在观看; bad inserts 它在观看 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它在观看一部电视剧。<br>Bad: 一部电视剧它在观看。 |
| multiple edits: bad deletes 它没吹; bad inserts 它没吹 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没吹什么双簧。<br>Bad: 什么双簧它没吹。 |
| multiple edits: bad deletes 它没弹; bad inserts 它没弹 | topicalization_OSV_mei | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 它没弹什么玻璃珠。<br>Bad: 什么玻璃珠它没弹。 |
| multiple edits: bad deletes 它没拍摄; bad inserts 它没拍摄 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没拍摄任何记录片。<br>Bad: 任何记录片它没拍摄。 |
| multiple edits: bad deletes 它没演奏; bad inserts 它没演奏 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没演奏任何华尔兹。<br>Bad: 任何华尔兹它没演奏。 |
| multiple edits: bad deletes 我们在吃; bad inserts 我们在吃 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在吃一桶方便面。<br>Bad: 一桶方便面我们在吃。 |
| multiple edits: bad deletes 我们在喝; bad inserts 我们在喝 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在喝一瓶冰红茶。<br>Bad: 一瓶冰红茶我们在喝。 |
| multiple edits: bad deletes 我们没拉; bad inserts 我们没拉 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没拉任何大提琴。<br>Bad: 任何大提琴我们没拉。 |
| multiple edits: bad deletes 我没吃; bad inserts 我没吃 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没吃任何葡萄。<br>Bad: 任何葡萄我没吃。 |
| multiple edits: bad deletes 我没喝; bad inserts 我没喝 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没喝什么葡萄汁。<br>Bad: 什么葡萄汁我没喝。 |
| multiple edits: bad deletes 我没开; bad inserts 我没开 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没开什么飞机。<br>Bad: 什么飞机我没开。 |
| multiple edits: bad deletes 我没演奏; bad inserts 我没演奏 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没演奏任何华尔兹。<br>Bad: 任何华尔兹我没演奏。 |
| multiple edits: bad inserts 一个橘子; bad deletes 一个橘子 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在吃一个橘子。<br>Bad: 一个橘子她们在吃。 |
| multiple edits: bad inserts 一个蛋糕; bad deletes 一个蛋糕 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在吃一个蛋糕。<br>Bad: 一个蛋糕她们在吃。 |
| multiple edits: bad inserts 一个馒头; bad deletes 一个馒头 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在吃一个馒头。<br>Bad: 一个馒头她们在吃。 |
| multiple edits: bad inserts 一条蛇; bad deletes 一条蛇 | topicalization_SOV | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 他在捕捉一条蛇。<br>Bad: 他一条蛇在捕捉。 |
| multiple edits: bad inserts 一杯红茶; bad deletes 一杯红茶 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在喝一杯红茶。<br>Bad: 一杯红茶她们在喝。 |
| multiple edits: bad inserts 一杯红酒; bad deletes 一杯红酒 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在喝一杯红酒。<br>Bad: 一杯红酒你们在喝。 |
| multiple edits: bad inserts 一瓶橙汁; bad deletes 一瓶橙汁 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们在喝一瓶橙汁。<br>Bad: 一瓶橙汁她们在喝。 |
| multiple edits: bad inserts 一部动画片; bad deletes 一部动画片 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们在拍摄一部动画片。<br>Bad: 一部动画片你们在拍摄。 |
| multiple edits: bad inserts 一部手账; bad deletes 一部手账 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们在看一部手账。<br>Bad: 一部手账他们在看。 |
| multiple edits: bad inserts 一部记录片; bad deletes 一部记录片 | topicalization_OSV | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们在拍摄一部记录片。<br>Bad: 一部记录片我们在拍摄。 |
| multiple edits: bad inserts 什么动作片; bad deletes 什么动作片 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没拍摄什么动作片。<br>Bad: 什么动作片她们没拍摄。 |
| multiple edits: bad inserts 什么动画片; bad deletes 什么动画片 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没制作什么动画片。<br>Bad: 什么动画片你们没制作。 |
| multiple edits: bad inserts 什么华尔兹; bad deletes 什么华尔兹 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没演奏什么华尔兹。<br>Bad: 什么华尔兹他们没演奏。 |
| multiple edits: bad inserts 什么戏曲; bad deletes 什么戏曲 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没唱什么戏曲。<br>Bad: 什么戏曲你们没唱。 |
| multiple edits: bad inserts 什么白酒; bad deletes 什么白酒 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没喝什么白酒。<br>Bad: 什么白酒她们没喝。 |
| multiple edits: bad inserts 什么红茶; bad deletes 什么红茶 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没喝什么红茶。<br>Bad: 什么红茶我们没喝。 |
| multiple edits: bad inserts 什么肚子; bad deletes 什么肚子 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没包扎什么肚子。<br>Bad: 什么肚子你们没包扎。 |
| multiple edits: bad inserts 什么腿; bad deletes 什么腿 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没打断什么腿。<br>Bad: 什么腿你没打断。 |
| multiple edits: bad inserts 什么蛇; bad deletes 什么蛇 | topicalization_OSV_mei | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 它没捕捉什么蛇。<br>Bad: 什么蛇它没捕捉。 |
| multiple edits: bad inserts 什么蛇; bad deletes 什么蛇 | topicalization_SOV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没捕捉什么蛇。<br>Bad: 她们什么蛇没捕捉。 |
| multiple edits: bad inserts 什么记录片; bad deletes 什么记录片 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没拍摄什么记录片。<br>Bad: 什么记录片你们没拍摄。 |
| multiple edits: bad inserts 什么面包; bad deletes 什么面包 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没吃什么面包。<br>Bad: 什么面包你们没吃。 |
| multiple edits: bad inserts 什么馒头; bad deletes 什么馒头 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没吃什么馒头。<br>Bad: 什么馒头她们没吃。 |
| multiple edits: bad inserts 任何动作片; bad deletes 任何动作片 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没拍摄任何动作片。<br>Bad: 任何动作片她们没拍摄。 |
| multiple edits: bad inserts 任何动画片; bad deletes 任何动画片 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没观看任何动画片。<br>Bad: 任何动画片我们没观看。 |
| multiple edits: bad inserts 任何卡车; bad deletes 任何卡车 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没驾驶任何卡车。<br>Bad: 任何卡车我没驾驶。 |
| multiple edits: bad inserts 任何啤酒; bad deletes 任何啤酒 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没喝任何啤酒。<br>Bad: 任何啤酒你们没喝。 |
| multiple edits: bad inserts 任何头; bad deletes 任何头 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没检查任何头。<br>Bad: 任何头你没检查。 |
| multiple edits: bad inserts 任何心脏; bad deletes 任何心脏 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你没检查任何心脏。<br>Bad: 任何心脏你没检查。 |
| multiple edits: bad inserts 任何戏曲; bad deletes 任何戏曲 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没唱任何戏曲。<br>Bad: 任何戏曲你们没唱。 |
| multiple edits: bad inserts 任何手; bad deletes 任何手 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没打断任何手。<br>Bad: 任何手他没打断。 |
| multiple edits: bad inserts 任何橘子; bad deletes 任何橘子 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没吃任何橘子。<br>Bad: 任何橘子他们没吃。 |
| multiple edits: bad inserts 任何歌; bad deletes 任何歌 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没唱任何歌。<br>Bad: 任何歌她们没唱。 |
| multiple edits: bad inserts 任何狂想曲; bad deletes 任何狂想曲 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没演奏任何狂想曲。<br>Bad: 任何狂想曲你们没演奏。 |
| multiple edits: bad inserts 任何电视剧; bad deletes 任何电视剧 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没制作任何电视剧。<br>Bad: 任何电视剧我们没制作。 |
| multiple edits: bad inserts 任何糖果; bad deletes 任何糖果 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没吃任何糖果。<br>Bad: 任何糖果她们没吃。 |
| multiple edits: bad inserts 任何美声; bad deletes 任何美声 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没唱任何美声。<br>Bad: 任何美声我们没唱。 |
| multiple edits: bad inserts 任何肚子; bad deletes 任何肚子 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没包扎任何肚子。<br>Bad: 任何肚子他们没包扎。 |
| multiple edits: bad inserts 任何胃; bad deletes 任何胃 | topicalization_SOV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 它没检查任何胃。<br>Bad: 它任何胃没检查。 |
| multiple edits: bad inserts 任何脚; bad deletes 任何脚 | topicalization_SOV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我没检查任何脚。<br>Bad: 我任何脚没检查。 |
| multiple edits: bad inserts 任何腿; bad deletes 任何腿 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没打断任何腿。<br>Bad: 任何腿他没打断。 |
| multiple edits: bad inserts 任何被子; bad deletes 任何被子 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没盖任何被子。<br>Bad: 任何被子他们没盖。 |
| multiple edits: bad inserts 任何视频; bad deletes 任何视频 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他没拍摄任何视频。<br>Bad: 任何视频他没拍摄。 |
| multiple edits: bad inserts 任何轮船; bad deletes 任何轮船 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她没驾驶任何轮船。<br>Bad: 任何轮船她没驾驶。 |
| multiple edits: bad inserts 任何钢琴; bad deletes 任何钢琴 | topicalization_OSV_mei | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没弹任何钢琴。<br>Bad: 任何钢琴你们没弹。 |

## verb_phrase

| Subtype / good-bad difference | UID | n | Chinese | Diacritic | Gap | Collapse | Example |
|---|---|---:|---:|---:|---:|---:|---|
| multiple edits: bad inserts 屠宰过; bad deletes 屠宰过 | left_adverbial_d | 12 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 你们的下属赤手空拳屠宰过牛。<br>Bad: 你们的下属屠宰过赤手空拳牛。 |
| multiple edits: bad deletes 花卷; bad inserts 花卷 | right_yijing_b | 2 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 陈大姐借给徐小姐花卷已经好几十次了。<br>Bad: 陈大姐借给徐小姐已经好几十次花卷了。 |
| multiple edits: bad deletes 可乐; bad inserts 可乐 | right_yijing_b | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 杨大哥借给你们可乐已经九次了。<br>Bad: 杨大哥借给你们已经九次可乐了。 |
| multiple edits: bad deletes 录像带; bad inserts 录像带 | right_yijing_b | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 小明借给她们录像带已经十几次了。<br>Bad: 小明借给她们已经十几次录像带了。 |
| multiple edits: bad deletes 红酒; bad inserts 红酒 | right_yijing_b | 1 | 0.0000 | 1.0000 | -1.0000 | 0.0000 | Good: 李先生递给他红酒已经四次了。<br>Bad: 李先生递给他已经四次红酒了。 |
| multiple edits: bad deletes 香蕉; bad inserts 香蕉 | right_yijing_b | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 徐小姐寄给刘先生香蕉已经非常多次了。<br>Bad: 徐小姐寄给刘先生已经非常多次香蕉了。 |
| multiple edits: 给 -> 过; bad deletes 小说; 了 -> 小说 | right_yijing_a | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 我寄给王小姐小说已经六次了。<br>Bad: 我寄过王小姐已经六次小说。 |
| multiple edits: 给 -> 过; bad deletes 矿泉水; 了 -> 矿泉水 | right_yijing_a | 1 | 1.0000 | 0.0000 | +1.0000 | 0.0000 | Good: 王大娘寄给你们矿泉水已经七次了。<br>Bad: 王大娘寄过你们已经七次矿泉水。 |
| multiple edits: bad inserts 小声; bad deletes 小声 | left_adverbial_negation | 9 | 1.0000 | 0.1111 | +0.8889 | 0.0000 | Good: 这九位母亲没有小声演奏奏鸣曲。<br>Bad: 这九位母亲小声没有演奏奏鸣曲。 |
| multiple edits: bad inserts 开过; bad deletes 开过 | left_adverbial_d | 6 | 1.0000 | 0.1667 | +0.8333 | 0.0000 | Good: 另外一个吉他手红着脸开过火车。<br>Bad: 另外一个吉他手开过红着脸火车。 |
| multiple edits: 对 -> 有点难过; bad deletes 有点难过 | adjective_transitive_dui | 6 | 1.0000 | 0.1667 | +0.8333 | 0.0000 | Good: 她对王姨的行为有点难过。<br>Bad: 她有点难过王姨的行为。 |
| multiple edits: 对 -> 比较难过; bad deletes 比较难过 | adjective_transitive_dui | 4 | 1.0000 | 0.2500 | +0.7500 | 0.0000 | Good: 他对王先生的表现比较难过。<br>Bad: 他比较难过王先生的表现。 |
| multiple edits: bad deletes 轻声; bad inserts 轻声 | left_adverbial_b | 7 | 0.8571 | 0.1429 | +0.7143 | 0.0000 | Good: 这七位舞者轻声跨越着海洋。<br>Bad: 这七位舞者跨越着轻声海洋。 |
| multiple edits: bad deletes 作业; bad inserts 作业 | right_yijing_b | 6 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 王姨借给我作业已经四次了。<br>Bad: 王姨借给我已经四次作业了。 |
| multiple edits: bad deletes 大声地; bad inserts 大声地 | left_adverbial_e | 6 | 1.0000 | 0.3333 | +0.6667 | 0.0000 | Good: 冯大哥大声地拿收音机给李四。<br>Bad: 冯大哥拿收音机大声地给李四。 |
| multiple edits: bad deletes 方便面; bad inserts 方便面 | right_yijing_b | 3 | 0.3333 | 1.0000 | -0.6667 | 0.0000 | Good: 他递给徐小姐方便面已经几次了。<br>Bad: 他递给徐小姐已经几次方便面了。 |
| multiple edits: bad inserts 拿作业; bad deletes 拿作业 | left_adverbial_e | 8 | 0.8750 | 0.2500 | +0.6250 | 0.0000 | Good: 张婶悄悄地拿作业给王五。<br>Bad: 张婶拿作业悄悄地给王五。 |
| multiple edits: bad inserts 制作过; bad deletes 制作过 | left_adverbial_d | 5 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 这位同事红着脸制作过手账。<br>Bad: 这位同事制作过红着脸手账。 |
| multiple edits: bad inserts 爆炒着; bad deletes 爆炒着 | left_adverbial_b | 5 | 1.0000 | 0.4000 | +0.6000 | 0.0000 | Good: 那六个记者红着脸爆炒着鸡。<br>Bad: 那六个记者爆炒着红着脸鸡。 |
| 以为 → 告知 | ya_insertion | 39 | 0.8718 | 0.2821 | +0.5897 | 0.0000 | Good: 他们以为呀，同事去弹玻璃珠了。<br>Bad: 他们告知呀，同事去弹玻璃珠了。 |
| multiple edits: bad inserts 麻醉着; bad deletes 麻醉着 | left_adverbial_b | 7 | 1.0000 | 0.4286 | +0.5714 | 0.0000 | Good: 你憋着气麻醉着大象。<br>Bad: 你麻醉着憋着气大象。 |
| multiple edits: bad inserts 麻醉过; bad deletes 麻醉过 | left_adverbial_d | 7 | 0.7143 | 0.1429 | +0.5714 | 0.0000 | Good: 陈大姐的儿子仰着头麻醉过大象。<br>Bad: 陈大姐的儿子麻醉过仰着头大象。 |
| multiple edits: bad inserts 爆炒过; bad deletes 爆炒过 | left_adverbial_d | 11 | 0.9091 | 0.3636 | +0.5455 | 0.0000 | Good: 我们举着手爆炒过鸡。<br>Bad: 我们爆炒过举着手鸡。 |
| multiple edits: bad inserts 跨越过; bad deletes 跨越过 | left_adverbial_d | 10 | 0.9000 | 0.4000 | +0.5000 | 0.0000 | Good: 这位上级光着膀子跨越过沙漠。<br>Bad: 这位上级跨越过光着膀子沙漠。 |
| multiple edits: bad deletes 啤酒; bad inserts 啤酒 | right_yijing_b | 6 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 李先生递给你们啤酒已经十几次了。<br>Bad: 李先生递给你们已经十几次啤酒了。 |
| multiple edits: bad deletes 开瓶器; bad inserts 开瓶器 | right_yijing_b | 6 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 他们借给我们开瓶器已经非常多次了。<br>Bad: 他们借给我们已经非常多次开瓶器了。 |
| multiple edits: bad inserts 捕捉过; bad deletes 捕捉过 | left_adverbial_d | 6 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 那位舞者赤手空拳捕捉过蛇。<br>Bad: 那位舞者捕捉过赤手空拳蛇。 |
| multiple edits: bad inserts 检查过; bad deletes 检查过 | left_adverbial_d | 6 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 这个妹妹红着脸检查过腿。<br>Bad: 这个妹妹检查过红着脸腿。 |
| multiple edits: bad inserts 拿开瓶器; bad deletes 拿开瓶器 | left_adverbial_e | 4 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 我们轻飘飘地拿开瓶器给他。<br>Bad: 我们拿开瓶器轻飘飘地给他。 |
| multiple edits: 给 -> 过; bad deletes 蛇; 了 -> 蛇 | right_yijing_a | 4 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 他们借给她蛇已经好几次了。<br>Bad: 他们借过她已经好几次蛇。 |
| multiple edits: bad deletes 咖啡; bad inserts 咖啡 | right_yijing_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 李先生送给赵大爷咖啡已经好几十次了。<br>Bad: 李先生送给赵大爷已经好几十次咖啡了。 |
| multiple edits: bad deletes 椅子; bad inserts 椅子 | right_yijing_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 徐小姐寄给你椅子已经三次了。<br>Bad: 徐小姐寄给你已经三次椅子了。 |
| multiple edits: bad deletes 橙汁; bad inserts 橙汁 | right_yijing_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 张先生借给冯大哥橙汁已经非常多次了。<br>Bad: 张先生借给冯大哥已经非常多次橙汁了。 |
| multiple edits: bad deletes 电视机; bad inserts 电视机 | right_yijing_b | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 他们送给我们电视机已经非常多次了。<br>Bad: 他们送给我们已经非常多次电视机了。 |
| multiple edits: bad deletes 白米饭; bad inserts 白米饭 | right_yijing_b | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 我们递给冯大哥白米饭已经三次了。<br>Bad: 我们递给冯大哥已经三次白米饭了。 |
| multiple edits: bad deletes 白酒; bad inserts 白酒 | right_yijing_b | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 她们寄给胡大爷白酒已经三次了。<br>Bad: 她们寄给胡大爷已经三次白酒了。 |
| multiple edits: bad deletes 缓缓; bad inserts 缓缓 | left_adverbial_d | 2 | 0.5000 | 0.0000 | +0.5000 | 0.0000 | Good: 那三个上级缓缓清蒸过鱼。<br>Bad: 那三个上级清蒸过缓缓鱼。 |
| multiple edits: bad deletes 被子; bad inserts 被子 | right_yijing_b | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 周大妈送给我被子已经八次了。<br>Bad: 周大妈送给我已经八次被子了。 |
| multiple edits: bad deletes 饮料瓶; bad inserts 饮料瓶 | right_yijing_b | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 郑大妈寄给你们饮料瓶已经好几十次了。<br>Bad: 郑大妈寄给你们已经好几十次饮料瓶了。 |
| multiple edits: bad inserts 先生把张; bad deletes 把张先生 | verb_phrase_left_negation | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 王先生没有把张先生安慰。<br>Bad: 王先生把张先生没有安慰。 |
| multiple edits: bad inserts 煮; bad deletes 煮了 | preposition_insertion | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 王大娘给我的学生煮了十几条鱼。<br>Bad: 王大娘煮给我的学生十几条鱼。 |
| multiple edits: 给 -> 过; bad deletes 充电器; 了 -> 充电器 | right_yijing_a | 2 | 0.5000 | 1.0000 | -0.5000 | 0.0000 | Good: 他借给她充电器已经八次了。<br>Bad: 他借过她已经八次充电器。 |
| multiple edits: 给 -> 过; bad deletes 蛋糕; 了 -> 蛋糕 | right_yijing_a | 2 | 1.0000 | 0.5000 | +0.5000 | 0.0000 | Good: 你送给你们蛋糕已经六次了。<br>Bad: 你送过你们已经六次蛋糕。 |
| multiple edits: bad inserts 屠宰着; bad deletes 屠宰着 | left_adverbial_b | 15 | 1.0000 | 0.5333 | +0.4667 | 0.0000 | Good: 刘先生的上级赤手空拳屠宰着牛。<br>Bad: 刘先生的上级屠宰着赤手空拳牛。 |
| multiple edits: bad deletes 鱼; bad inserts 鱼 | right_yijing_b | 38 | 0.4211 | 0.8684 | -0.4474 | 0.0000 | Good: 陈大姐寄给宋女士鱼已经十几次了。<br>Bad: 陈大姐寄给宋女士已经十几次鱼了。 |
| multiple edits: bad inserts 拿教材; bad deletes 拿教材 | left_adverbial_e | 18 | 0.9444 | 0.5000 | +0.4444 | 0.0000 | Good: 你慢吞吞地拿教材给王大娘。<br>Bad: 你拿教材慢吞吞地给王大娘。 |
| multiple edits: bad deletes 轻声; bad inserts 轻声 | left_adverbial_d | 9 | 0.8889 | 0.4444 | +0.4444 | 0.0000 | Good: 这个领导轻声爆炒过鸡。<br>Bad: 这个领导爆炒过轻声鸡。 |
| multiple edits: bad inserts 悄悄; bad deletes 悄悄 | left_adverbial_negation | 9 | 0.8889 | 0.4444 | +0.4444 | 0.0000 | Good: 王五的老板没有悄悄拉大提琴。<br>Bad: 王五的老板悄悄没有拉大提琴。 |
| multiple edits: bad inserts 炖着; bad deletes 炖着 | left_adverbial_b | 17 | 0.9412 | 0.5294 | +0.4118 | 0.0000 | Good: 她举着手炖着鸭。<br>Bad: 她炖着举着手鸭。 |
| multiple edits: bad deletes 静静地; bad inserts 静静地 | left_adverbial_e | 5 | 0.6000 | 1.0000 | -0.4000 | 0.0000 | Good: 他们静静地拿热水器给胡大爷。<br>Bad: 他们拿热水器静静地给胡大爷。 |
| multiple edits: bad inserts 拿饮料瓶; bad deletes 拿饮料瓶 | left_adverbial_e | 5 | 0.4000 | 0.0000 | +0.4000 | 0.0000 | Good: 她们大摇大摆地拿饮料瓶给冯大哥。<br>Bad: 她们拿饮料瓶大摇大摆地给冯大哥。 |
| multiple edits: 对 -> 有点高兴; bad deletes 有点高兴 | adjective_transitive_dui | 5 | 0.8000 | 0.4000 | +0.4000 | 0.0000 | Good: 她对郑大妈的表现有点高兴。<br>Bad: 她有点高兴郑大妈的表现。 |
| multiple edits: 给 -> 过; bad deletes 饮料瓶; 了 -> 饮料瓶 | right_yijing_a | 5 | 1.0000 | 0.6000 | +0.4000 | 0.0000 | Good: 她们借给小明饮料瓶已经好几百次了。<br>Bad: 她们借过小明已经好几百次饮料瓶。 |
| multiple edits: bad deletes 书; bad inserts 书 | right_yijing_b | 5 | 0.6000 | 0.2000 | +0.4000 | 0.0000 | Good: 徐小姐借给张先生书已经好几十次了。<br>Bad: 徐小姐借给张先生已经好几十次书了。 |
| multiple edits: bad inserts 静静; bad deletes 静静 | left_adverbial_negation | 21 | 0.8095 | 0.4286 | +0.3810 | 0.0000 | Good: 这个罪犯没有静静麻醉大象。<br>Bad: 这个罪犯静静没有麻醉大象。 |
| multiple edits: 给 -> 过; bad deletes 鸭; 了 -> 鸭 | right_yijing_a | 24 | 1.0000 | 0.6250 | +0.3750 | 0.0000 | Good: 杨大哥寄给我们鸭已经七次了。<br>Bad: 杨大哥寄过我们已经七次鸭。 |
| multiple edits: 对 -> 非常高兴; bad deletes 非常高兴 | adjective_transitive_dui | 8 | 0.6250 | 1.0000 | -0.3750 | 0.0000 | Good: 他们对李先生的所作所为非常高兴。<br>Bad: 他们非常高兴李先生的所作所为。 |
| multiple edits: bad inserts 包扎; bad deletes 包扎了 | preposition_insertion | 11 | 1.0000 | 0.6364 | +0.3636 | 0.0000 | Good: 王姨给那四位钢琴家包扎了三个耳朵。<br>Bad: 王姨包扎给那四位钢琴家三个耳朵。 |
| multiple edits: bad inserts 被; bad deletes 被 | verb_phrase_left_adverbial | 300 | 0.8667 | 0.5100 | +0.3567 | 0.0000 | Good: 王五当时被张夫人约束了。<br>Bad: 王五被当时张夫人约束了。 |
| multiple edits: bad inserts 烧过; bad deletes 烧过 | left_adverbial_d | 17 | 1.0000 | 0.6471 | +0.3529 | 0.0000 | Good: 她们的女儿悄悄烧过鱼。<br>Bad: 她们的女儿烧过悄悄鱼。 |
| multiple edits: bad inserts 煮着; bad deletes 煮着 | left_adverbial_b | 20 | 1.0000 | 0.6500 | +0.3500 | 0.0000 | Good: 你们努力煮着鸡。<br>Bad: 你们煮着努力鸡。 |
| multiple edits: bad deletes 鸭; bad inserts 鸭 | right_yijing_b | 29 | 0.8966 | 0.5517 | +0.3448 | 0.0000 | Good: 他们寄给徐小姐鸭已经七次了。<br>Bad: 他们寄给徐小姐已经七次鸭了。 |
| multiple edits: bad inserts 烧着; bad deletes 烧着 | left_adverbial_b | 9 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 李四的学生急速烧着鱼。<br>Bad: 李四的学生烧着急速鱼。 |
| multiple edits: bad inserts 煮过; bad deletes 煮过 | left_adverbial_d | 9 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 她们红着脸煮过鸡。<br>Bad: 她们煮过红着脸鸡。 |
| multiple edits: bad inserts 制作; bad deletes 制作了 | preposition_insertion | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 刘先生给她的爸爸制作了好几十部动画片。<br>Bad: 刘先生制作给她的爸爸好几十部动画片。 |
| multiple edits: bad inserts 观看着; bad deletes 观看着 | left_adverbial_b | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这个妹妹憋着气观看着电影。<br>Bad: 这个妹妹观看着憋着气电影。 |
| multiple edits: 对 -> 有点沉默; bad deletes 有点沉默 | adjective_transitive_dui | 6 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 她对王姨的所作所为有点沉默。<br>Bad: 她有点沉默王姨的所作所为。 |
| multiple edits: bad deletes 裙子; bad inserts 裙子 | right_yijing_b | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 他们递给王姨裙子已经十次了。<br>Bad: 他们递给王姨已经十次裙子了。 |
| multiple edits: bad inserts 包扎着; bad deletes 包扎着 | left_adverbial_b | 3 | 1.0000 | 0.6667 | +0.3333 | 0.0000 | Good: 这个姐姐鼓起勇气包扎着手。<br>Bad: 这个姐姐包扎着鼓起勇气手。 |
| multiple edits: bad inserts 清蒸过; bad deletes 清蒸过 | left_adverbial_d | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 张三低着头清蒸过鱼。<br>Bad: 张三清蒸过低着头鱼。 |
| multiple edits: bad inserts 演奏着; bad deletes 演奏着 | left_adverbial_b | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 杨大哥空着手演奏着协奏曲。<br>Bad: 杨大哥演奏着空着手协奏曲。 |
| multiple edits: bad inserts 领养着; bad deletes 领养着 | left_adverbial_b | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 另外五位员工咬着牙领养着小猫。<br>Bad: 另外五位员工领养着咬着牙小猫。 |
| multiple edits: 对 -> 比较快乐; bad deletes 比较快乐 | adjective_transitive_dui | 3 | 0.6667 | 1.0000 | -0.3333 | 0.0000 | Good: 他们对小明的表现比较快乐。<br>Bad: 他们比较快乐小明的表现。 |
| multiple edits: bad deletes 急速; bad inserts 急速 | left_adverbial_d | 6 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 赵大爷的儿子急速包扎过脚。<br>Bad: 赵大爷的儿子包扎过急速脚。 |
| multiple edits: bad deletes 热水器; bad inserts 热水器 | right_yijing_b | 3 | 0.3333 | 0.0000 | +0.3333 | 0.0000 | Good: 你借给你们热水器已经八次了。<br>Bad: 你借给你们已经八次热水器了。 |
| multiple edits: bad deletes 矿泉水; bad inserts 矿泉水 | right_yijing_b | 3 | 0.6667 | 0.3333 | +0.3333 | 0.0000 | Good: 她借给张先生矿泉水已经几次了。<br>Bad: 她借给张先生已经几次矿泉水了。 |
| multiple edits: bad inserts 拿杯子; bad deletes 拿杯子 | left_adverbial_e | 28 | 1.0000 | 0.6786 | +0.3214 | 0.0000 | Good: 我们慢吞吞地拿杯子给赵大爷。<br>Bad: 我们拿杯子慢吞吞地给赵大爷。 |
| multiple edits: bad inserts 捕捉; bad deletes 捕捉了 | preposition_insertion | 20 | 1.0000 | 0.7000 | +0.3000 | 0.0000 | Good: 张夫人给这个顾客捕捉了十头大象。<br>Bad: 张夫人捕捉给这个顾客十头大象。 |
| multiple edits: 对 -> 比较困惑; bad deletes 比较困惑 | adjective_transitive_dui | 10 | 1.0000 | 0.7000 | +0.3000 | 0.0000 | Good: 他对王大娘的行为比较困惑。<br>Bad: 他比较困惑王大娘的行为。 |
| multiple edits: 给 -> 过; bad deletes 手套; 了 -> 手套 | right_yijing_a | 14 | 1.0000 | 0.7143 | +0.2857 | 0.0000 | Good: 张婶借给李先生手套已经好几十次了。<br>Bad: 张婶借过李先生已经好几十次手套。 |
| multiple edits: bad deletes 偷偷; bad inserts 偷偷 | left_adverbial_b | 7 | 1.0000 | 0.7143 | +0.2857 | 0.0000 | Good: 那个小孩偷偷检查着肚子。<br>Bad: 那个小孩检查着偷偷肚子。 |
| multiple edits: bad deletes 小狗; bad inserts 小狗 | right_yijing_b | 7 | 0.8571 | 0.5714 | +0.2857 | 0.0000 | Good: 你送给你们小狗已经十几次了。<br>Bad: 你送给你们已经十几次小狗了。 |
| multiple edits: bad deletes 静静; bad inserts 静静 | left_adverbial_d | 7 | 0.4286 | 0.1429 | +0.2857 | 0.0000 | Good: 他静静演奏过狂想曲。<br>Bad: 他演奏过静静狂想曲。 |
| multiple edits: bad inserts 观看过; bad deletes 观看过 | left_adverbial_d | 7 | 1.0000 | 0.7143 | +0.2857 | 0.0000 | Good: 你低着头观看过动画片。<br>Bad: 你观看过低着头动画片。 |
| multiple edits: bad inserts 驾驶过; bad deletes 驾驶过 | left_adverbial_d | 7 | 0.8571 | 0.5714 | +0.2857 | 0.0000 | Good: 这八位学生甩着胳膊驾驶过卡车。<br>Bad: 这八位学生驾驶过甩着胳膊卡车。 |
| multiple edits: 对 -> 非常悲伤; bad deletes 非常悲伤 | adjective_transitive_dui | 7 | 1.0000 | 0.7143 | +0.2857 | 0.0000 | Good: 她对杨大哥的行为非常悲伤。<br>Bad: 她非常悲伤杨大哥的行为。 |
| multiple edits: bad inserts 慢慢; bad deletes 慢慢 | left_adverbial_negation | 11 | 1.0000 | 0.7273 | +0.2727 | 0.0000 | Good: 这位上级没有慢慢创作小说。<br>Bad: 这位上级慢慢没有创作小说。 |
| multiple edits: bad inserts 缓缓; bad deletes 缓缓 | left_adverbial_negation | 11 | 0.9091 | 0.6364 | +0.2727 | 0.0000 | Good: 王大娘的哥哥没有缓缓炖鸡。<br>Bad: 王大娘的哥哥缓缓没有炖鸡。 |
| multiple edits: bad inserts 创作着; bad deletes 创作着 | left_adverbial_b | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 他甩着胳膊创作着小说。<br>Bad: 他创作着甩着胳膊小说。 |
| multiple edits: bad inserts 拿被子; bad deletes 拿被子 | left_adverbial_e | 8 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 陈大姐轻轻地拿被子给杨大哥。<br>Bad: 陈大姐拿被子轻轻地给杨大哥。 |
| multiple edits: 对 -> 很悲伤; bad deletes 很悲伤 | adjective_transitive_dui | 8 | 0.8750 | 0.6250 | +0.2500 | 0.0000 | Good: 她对何太太的所作所为很悲伤。<br>Bad: 她很悲伤何太太的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 开瓶器; 了 -> 开瓶器 | right_yijing_a | 8 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 冯大哥送给我开瓶器已经十次了。<br>Bad: 冯大哥送过我已经十次开瓶器。 |
| multiple edits: bad deletes 充电器; bad inserts 充电器 | right_yijing_b | 4 | 0.0000 | 0.2500 | -0.2500 | 0.0000 | Good: 王五递给你充电器已经六次了。<br>Bad: 王五递给你已经六次充电器了。 |
| multiple edits: bad deletes 小声; bad inserts 小声 | left_adverbial_d | 4 | 0.2500 | 0.5000 | -0.2500 | 0.0000 | Good: 那个记者小声演奏过狂想曲。<br>Bad: 那个记者演奏过小声狂想曲。 |
| multiple edits: bad deletes 急速; bad inserts 急速 | left_adverbial_b | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 那九个奴隶急速制作着视频。<br>Bad: 那九个奴隶制作着急速视频。 |
| multiple edits: bad deletes 糖果; bad inserts 糖果 | right_yijing_b | 4 | 0.7500 | 0.5000 | +0.2500 | 0.0000 | Good: 张夫人寄给我们糖果已经八次了。<br>Bad: 张夫人寄给我们已经八次糖果了。 |
| multiple edits: bad inserts 弹; bad deletes 弹了 | preposition_insertion | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 李太太给这位领导弹了一个玻璃珠。<br>Bad: 李太太弹给这位领导一个玻璃珠。 |
| multiple edits: bad inserts 拿充电器; bad deletes 拿充电器 | left_adverbial_e | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 她大摇大摆地拿充电器给她们。<br>Bad: 她拿充电器大摇大摆地给她们。 |
| multiple edits: bad inserts 拿电视机; bad deletes 拿电视机 | left_adverbial_e | 4 | 0.5000 | 0.7500 | -0.2500 | 0.0000 | Good: 赵大爷轻飘飘地拿电视机给张婶。<br>Bad: 赵大爷拿电视机轻飘飘地给张婶。 |
| multiple edits: bad inserts 盖; bad deletes 盖了 | preposition_insertion | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 李四给另外九位空姐盖了九条被子。<br>Bad: 李四盖给另外九位空姐九条被子。 |
| multiple edits: 对 -> 有点困惑; bad deletes 有点困惑 | adjective_transitive_dui | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 她们对何太太的所作所为有点困惑。<br>Bad: 她们有点困惑何太太的所作所为。 |
| multiple edits: 对 -> 非常开心; bad deletes 非常开心 | adjective_transitive_dui | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 他对宋女士的表现非常开心。<br>Bad: 他非常开心宋女士的表现。 |
| multiple edits: 给 -> 过; bad deletes 小狗; 了 -> 小狗 | right_yijing_a | 4 | 1.0000 | 0.7500 | +0.2500 | 0.0000 | Good: 我递给你们小狗已经九次了。<br>Bad: 我递过你们已经九次小狗。 |
| multiple edits: 给 -> 过; bad deletes 白酒; 了 -> 白酒 | right_yijing_a | 4 | 0.7500 | 1.0000 | -0.2500 | 0.0000 | Good: 她送给小明白酒已经五次了。<br>Bad: 她送过小明已经五次白酒。 |
| multiple edits: 给 -> 过; bad deletes 老虎; 了 -> 老虎 | right_yijing_a | 18 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 周大妈借给李先生老虎已经七次了。<br>Bad: 周大妈借过李先生已经七次老虎。 |
| multiple edits: bad inserts 预习; bad deletes 预习了 | preposition_insertion | 9 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 张夫人给王小姐的上级预习了八本教材。<br>Bad: 张夫人预习给王小姐的上级八本教材。 |
| multiple edits: 对 -> 有点伤心; bad deletes 有点伤心 | adjective_transitive_dui | 9 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 她对刘先生的表现有点伤心。<br>Bad: 她有点伤心刘先生的表现。 |
| multiple edits: 对 -> 比较苦恼; bad deletes 比较苦恼 | adjective_transitive_dui | 9 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 他对郑大妈的所作所为比较苦恼。<br>Bad: 他比较苦恼郑大妈的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 牛; 了 -> 牛 | right_yijing_a | 9 | 1.0000 | 0.7778 | +0.2222 | 0.0000 | Good: 王姨寄给我们牛已经许多次了。<br>Bad: 王姨寄过我们已经许多次牛。 |
| multiple edits: bad inserts 拿袜子; bad deletes 拿袜子 | left_adverbial_e | 33 | 1.0000 | 0.7879 | +0.2121 | 0.0000 | Good: 冯大哥小心地拿袜子给张三。<br>Bad: 冯大哥拿袜子小心地给张三。 |
| multiple edits: bad deletes 没有; bad inserts 没有 | verb_phrase_left_negation | 224 | 0.9018 | 0.7009 | +0.2009 | 0.0000 | Good: 王姨没有把杨大哥批评。<br>Bad: 王姨把杨大哥没有批评。 |
| multiple edits: bad deletes 牛; bad inserts 牛 | right_yijing_b | 10 | 0.9000 | 0.7000 | +0.2000 | 0.0000 | Good: 张三送给赵大爷牛已经三次了。<br>Bad: 张三送给赵大爷已经三次牛了。 |
| multiple edits: bad deletes 大声; bad inserts 大声 | left_adverbial_b | 5 | 0.8000 | 0.6000 | +0.2000 | 0.0000 | Good: 他们大声驾驶着轮船。<br>Bad: 他们驾驶着大声轮船。 |
| multiple edits: bad inserts 炖; bad deletes 炖了 | preposition_insertion | 5 | 0.8000 | 0.6000 | +0.2000 | 0.0000 | Good: 小王给那位老板炖了十几条鱼。<br>Bad: 小王炖给那位老板十几条鱼。 |
| multiple edits: bad inserts 看过; bad deletes 看过 | left_adverbial_d | 10 | 0.5000 | 0.3000 | +0.2000 | 0.0000 | Good: 那个姐姐空着手看过教材。<br>Bad: 那个姐姐看过空着手教材。 |
| multiple edits: bad deletes 手套; bad inserts 手套 | right_yijing_b | 10 | 0.4000 | 0.6000 | -0.2000 | 0.0000 | Good: 李太太寄给他手套已经八次了。<br>Bad: 李太太寄给他已经八次手套了。 |
| multiple edits: bad inserts 拍摄着; bad deletes 拍摄着 | left_adverbial_b | 10 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 宋女士闭着眼睛拍摄着电影。<br>Bad: 宋女士拍摄着闭着眼睛电影。 |
| multiple edits: bad inserts 炖过; bad deletes 炖过 | left_adverbial_d | 10 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 这位打工人咧着嘴炖过鱼。<br>Bad: 这位打工人炖过咧着嘴鱼。 |
| multiple edits: bad deletes 小声; bad inserts 小声 | left_adverbial_b | 5 | 0.4000 | 0.6000 | -0.2000 | 0.0000 | Good: 吴太太小声创作着漫画。<br>Bad: 吴太太创作着小声漫画。 |
| multiple edits: bad deletes 小声地; bad inserts 小声地 | left_adverbial_e | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 你小声地拿开瓶器给王先生。<br>Bad: 你拿开瓶器小声地给王先生。 |
| multiple edits: bad deletes 慢慢; bad inserts 慢慢 | left_adverbial_d | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 这位上级慢慢拍摄过电影。<br>Bad: 这位上级拍摄过慢慢电影。 |
| multiple edits: bad inserts 们把他; bad deletes 把他们 | verb_phrase_left_negation | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 你们没有把他们奖励。<br>Bad: 你们把他们没有奖励。 |
| multiple edits: bad inserts 拿玻璃珠; bad deletes 拿玻璃珠 | left_adverbial_e | 5 | 1.0000 | 0.8000 | +0.2000 | 0.0000 | Good: 他慢吞吞地拿玻璃珠给小王。<br>Bad: 他拿玻璃珠慢吞吞地给小王。 |
| multiple edits: 对 -> 比较伤心; bad deletes 比较伤心 | adjective_transitive_dui | 5 | 0.8000 | 1.0000 | -0.2000 | 0.0000 | Good: 他们对胡大爷的行为比较伤心。<br>Bad: 他们比较伤心胡大爷的行为。 |
| multiple edits: bad deletes 老虎; bad inserts 老虎 | right_yijing_b | 16 | 0.6250 | 0.8125 | -0.1875 | 0.0000 | Good: 周大妈寄给郑大妈老虎已经三次了。<br>Bad: 周大妈寄给郑大妈已经三次老虎了。 |
| multiple edits: bad deletes 偷偷; bad inserts 偷偷 | left_adverbial_d | 11 | 1.0000 | 0.8182 | +0.1818 | 0.0000 | Good: 小王偷偷跨越过沙漠。<br>Bad: 小王跨越过偷偷沙漠。 |
| 过 → 了 | verb_negation_particle | 300 | 0.8667 | 0.6900 | +0.1767 | 0.0000 | Good: 她没有清蒸过鸡。<br>Bad: 她没有清蒸了鸡。 |
| multiple edits: bad deletes 都; bad inserts 都 | left_dou | 300 | 0.9733 | 0.8033 | +0.1700 | 0.0000 | Good: 我们都闭着眼睛地吃着糖。<br>Bad: 我们闭着眼睛地都吃着糖。 |
| multiple edits: bad deletes 蛇; bad inserts 蛇 | right_yijing_b | 12 | 0.6667 | 0.8333 | -0.1667 | 0.0000 | Good: 李四寄给她蛇已经三次了。<br>Bad: 李四寄给她已经三次蛇了。 |
| multiple edits: bad deletes 教材; bad inserts 教材 | right_yijing_b | 6 | 0.8333 | 0.6667 | +0.1667 | 0.0000 | Good: 他们送给她教材已经许多次了。<br>Bad: 他们送给她已经许多次教材了。 |
| multiple edits: 对 -> 很快乐; bad deletes 很快乐 | adjective_transitive_dui | 6 | 0.6667 | 0.8333 | -0.1667 | 0.0000 | Good: 他们对吴太太的行为很快乐。<br>Bad: 他们很快乐吴太太的行为。 |
| multiple edits: bad inserts 包扎过; bad deletes 包扎过 | left_adverbial_d | 6 | 0.5000 | 0.3333 | +0.1667 | 0.0000 | Good: 她们举着手包扎过手。<br>Bad: 她们包扎过举着手手。 |
| multiple edits: bad deletes 放心地; bad inserts 放心地 | left_adverbial_e | 12 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 宋女士放心地拿录像带给李先生。<br>Bad: 宋女士拿录像带放心地给李先生。 |
| multiple edits: bad inserts 创作过; bad deletes 创作过 | left_adverbial_d | 12 | 0.6667 | 0.5000 | +0.1667 | 0.0000 | Good: 这八个罪犯举着手创作过小说。<br>Bad: 这八个罪犯创作过举着手小说。 |
| multiple edits: bad inserts 吃着; bad deletes 吃着 | left_adverbial_b | 12 | 0.9167 | 0.7500 | +0.1667 | 0.0000 | Good: 郑大妈的领导嘟着嘴吃着方便面。<br>Bad: 郑大妈的领导吃着嘟着嘴方便面。 |
| multiple edits: 给 -> 过; bad deletes 袜子; 了 -> 袜子 | right_yijing_a | 12 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 张三递给王小姐袜子已经许多次了。<br>Bad: 张三递过王小姐已经许多次袜子。 |
| multiple edits: bad deletes 努力地; bad inserts 努力地 | left_adverbial_e | 6 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 你努力地拿饮料瓶给我们。<br>Bad: 你拿饮料瓶努力地给我们。 |
| multiple edits: bad deletes 奋力地; bad inserts 奋力地 | left_adverbial_e | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 张婶奋力地拿电视机给徐小姐。<br>Bad: 张婶拿电视机奋力地给徐小姐。 |
| multiple edits: bad deletes 悄悄; bad inserts 悄悄 | left_adverbial_b | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 这个奴隶悄悄麻醉着老虎。<br>Bad: 这个奴隶麻醉着悄悄老虎。 |
| multiple edits: bad inserts 驾驶着; bad deletes 驾驶着 | left_adverbial_b | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 你红着脸驾驶着货车。<br>Bad: 你驾驶着红着脸货车。 |
| multiple edits: bad inserts 麻醉; bad deletes 麻醉了 | preposition_insertion | 6 | 0.8333 | 1.0000 | -0.1667 | 0.0000 | Good: 张三给李太太的兄弟麻醉了许多头大象。<br>Bad: 张三麻醉给李太太的兄弟许多头大象。 |
| multiple edits: 对 -> 有点悲伤; bad deletes 有点悲伤 | adjective_transitive_dui | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 她对吴太太的表现有点悲伤。<br>Bad: 她有点悲伤吴太太的表现。 |
| multiple edits: 对 -> 非常沉默; bad deletes 非常沉默 | adjective_transitive_dui | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 他对刘先生的表现非常沉默。<br>Bad: 他非常沉默刘先生的表现。 |
| multiple edits: 给 -> 过; bad deletes 杯子; 了 -> 杯子 | right_yijing_a | 6 | 1.0000 | 0.8333 | +0.1667 | 0.0000 | Good: 我寄给宋女士杯子已经非常多次了。<br>Bad: 我寄过宋女士已经非常多次杯子。 |
| multiple edits: bad inserts 制作着; bad deletes 制作着 | left_adverbial_b | 13 | 0.8462 | 0.6923 | +0.1538 | 0.0000 | Good: 我们的哥哥空着手制作着动画片。<br>Bad: 我们的哥哥制作着空着手动画片。 |
| multiple edits: bad inserts 喝着; bad deletes 喝着 | left_adverbial_b | 13 | 0.9231 | 0.7692 | +0.1538 | 0.0000 | Good: 他轻声喝着矿泉水。<br>Bad: 他喝着轻声矿泉水。 |
| multiple edits: bad deletes 鸡; bad inserts 鸡 | right_yijing_b | 21 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 王五送给张先生鸡已经几次了。<br>Bad: 王五送给张先生已经几次鸡了。 |
| multiple edits: bad inserts 弹过; bad deletes 弹过 | left_adverbial_d | 14 | 0.7143 | 0.5714 | +0.1429 | 0.0000 | Good: 李太太的员工低着头弹过玻璃珠。<br>Bad: 李太太的员工弹过低着头玻璃珠。 |
| multiple edits: bad deletes 悄悄; bad inserts 悄悄 | left_adverbial_d | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 张先生的女儿悄悄检查过眼睛。<br>Bad: 张先生的女儿检查过悄悄眼睛。 |
| multiple edits: 对 -> 很难过; bad deletes 很难过 | adjective_transitive_dui | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 她对郑大妈的表现很难过。<br>Bad: 她很难过郑大妈的表现。 |
| multiple edits: 对 -> 很高兴; bad deletes 很高兴 | adjective_transitive_dui | 7 | 0.5714 | 0.7143 | -0.1429 | 0.0000 | Good: 他对胡大爷的所作所为很高兴。<br>Bad: 他很高兴胡大爷的所作所为。 |
| multiple edits: 对 -> 非常冷静; bad deletes 非常冷静 | adjective_transitive_dui | 7 | 1.0000 | 0.8571 | +0.1429 | 0.0000 | Good: 她对刘先生的行为非常冷静。<br>Bad: 她非常冷静刘先生的行为。 |
| multiple edits: bad deletes 袜子; bad inserts 袜子 | right_yijing_b | 21 | 0.7619 | 0.6190 | +0.1429 | 0.0000 | Good: 你们递给刘先生袜子已经八次了。<br>Bad: 你们递给刘先生已经八次袜子了。 |
| multiple edits: bad deletes 衣服; bad inserts 衣服 | right_yijing_b | 7 | 0.7143 | 0.8571 | -0.1429 | 0.0000 | Good: 吴太太递给她们衣服已经两次了。<br>Bad: 吴太太递给她们已经两次衣服了。 |
| multiple edits: bad deletes 没有; bad inserts 没有 | left_adverbial_negation | 157 | 0.7580 | 0.6178 | +0.1401 | 0.0000 | Good: 另外三位顾客没有低着头喝咖啡。<br>Bad: 另外三位顾客低着头没有喝咖啡。 |
| multiple edits: 给 -> 过; bad deletes 鸡; 了 -> 鸡 | right_yijing_a | 32 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 她送给我们鸡已经两次了。<br>Bad: 她送过我们已经两次鸡。 |
| multiple edits: bad deletes 专心; bad inserts 专心 | left_adverbial_b | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 她们的哥哥专心麻醉着老虎。<br>Bad: 她们的哥哥麻醉着专心老虎。 |
| multiple edits: bad deletes 大声; bad inserts 大声 | left_adverbial_d | 8 | 0.7500 | 0.8750 | -0.1250 | 0.0000 | Good: 张先生大声创作过小说。<br>Bad: 张先生创作过大声小说。 |
| multiple edits: bad deletes 慢慢; bad inserts 慢慢 | left_adverbial_b | 8 | 1.0000 | 0.8750 | +0.1250 | 0.0000 | Good: 这位员工慢慢清蒸着鱼。<br>Bad: 这位员工清蒸着慢慢鱼。 |
| multiple edits: bad inserts 大声; bad deletes 大声 | left_adverbial_negation | 8 | 0.7500 | 0.8750 | -0.1250 | 0.0000 | Good: 他没有大声买开瓶器。<br>Bad: 他大声没有买开瓶器。 |
| multiple edits: bad inserts 捕捉着; bad deletes 捕捉着 | left_adverbial_b | 8 | 0.8750 | 0.7500 | +0.1250 | 0.0000 | Good: 他的儿子空着手捕捉着鸭。<br>Bad: 他的儿子捕捉着空着手鸭。 |
| multiple edits: 对 -> 有点苦恼; bad deletes 有点苦恼 | adjective_transitive_dui | 8 | 0.8750 | 0.7500 | +0.1250 | 0.0000 | Good: 她对何太太的所作所为有点苦恼。<br>Bad: 她有点苦恼何太太的所作所为。 |
| multiple edits: bad inserts 拿衣服; bad deletes 拿衣服 | left_adverbial_e | 17 | 0.5882 | 0.4706 | +0.1176 | 0.0000 | Good: 你们大声地拿衣服给你。<br>Bad: 你们拿衣服大声地给你。 |
| multiple edits: bad deletes 缓缓; bad inserts 缓缓 | left_adverbial_b | 9 | 0.6667 | 0.7778 | -0.1111 | 0.0000 | Good: 这八位服务员缓缓演奏着奏鸣曲。<br>Bad: 这八位服务员演奏着缓缓奏鸣曲。 |
| multiple edits: bad deletes 静静; bad inserts 静静 | left_adverbial_b | 9 | 0.6667 | 0.7778 | -0.1111 | 0.0000 | Good: 张三静静创作着小说。<br>Bad: 张三创作着静静小说。 |
| multiple edits: bad inserts 跨越着; bad deletes 跨越着 | left_adverbial_b | 9 | 0.7778 | 0.6667 | +0.1111 | 0.0000 | Good: 那六个姐姐嘟着嘴跨越着沙漠。<br>Bad: 那六个姐姐跨越着嘟着嘴沙漠。 |
| multiple edits: bad inserts 开着; bad deletes 开着 | left_adverbial_b | 9 | 0.6667 | 0.5556 | +0.1111 | 0.0000 | Good: 张三红着脸开着卡车。<br>Bad: 张三开着红着脸卡车。 |
| multiple edits: bad inserts 急速; bad deletes 急速 | left_adverbial_negation | 9 | 0.8889 | 0.7778 | +0.1111 | 0.0000 | Good: 另外五个姐姐没有急速跨越沙漠。<br>Bad: 另外五个姐姐急速没有跨越沙漠。 |
| multiple edits: bad inserts 拉着; bad deletes 拉着 | left_adverbial_b | 9 | 0.7778 | 0.8889 | -0.1111 | 0.0000 | Good: 我们的同事咬着牙拉着小提琴。<br>Bad: 我们的同事拉着咬着牙小提琴。 |
| multiple edits: bad inserts 领养过; bad deletes 领养过 | left_adverbial_d | 9 | 0.5556 | 0.6667 | -0.1111 | 0.0000 | Good: 她嘟着嘴领养过小猫。<br>Bad: 她领养过嘟着嘴小猫。 |
| multiple edits: 对 -> 非常苦恼; bad deletes 非常苦恼 | adjective_transitive_dui | 9 | 0.8889 | 0.7778 | +0.1111 | 0.0000 | Good: 他对郑大妈的行为非常苦恼。<br>Bad: 他非常苦恼郑大妈的行为。 |
| multiple edits: bad inserts 拉过; bad deletes 拉过 | left_adverbial_d | 10 | 0.2000 | 0.1000 | +0.1000 | 0.0000 | Good: 那个上级举着手拉过小提琴。<br>Bad: 那个上级拉过举着手小提琴。 |
| multiple edits: bad inserts 唱着; bad deletes 唱着 | left_adverbial_b | 10 | 1.0000 | 0.9000 | +0.1000 | 0.0000 | Good: 那两位工人光着膀子唱着小调。<br>Bad: 那两位工人唱着光着膀子小调。 |
| multiple edits: 对 -> 有点愤怒; bad deletes 有点愤怒 | adjective_transitive_dui | 10 | 1.0000 | 0.9000 | +0.1000 | 0.0000 | Good: 她们对杨大哥的所作所为有点愤怒。<br>Bad: 她们有点愤怒杨大哥的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 裙子; 了 -> 裙子 | right_yijing_a | 10 | 1.0000 | 0.9000 | +0.1000 | 0.0000 | Good: 冯大哥借给郑大妈裙子已经两次了。<br>Bad: 冯大哥借过郑大妈已经两次裙子。 |
| 觉得 → 告知 | ya_insertion | 43 | 0.6512 | 0.7442 | -0.0930 | 0.0000 | Good: 张三觉得呀，服务员去看书了。<br>Bad: 张三告知呀，服务员去看书了。 |
| multiple edits: bad inserts 弹着; bad deletes 弹着 | left_adverbial_b | 11 | 0.7273 | 0.8182 | -0.0909 | 0.0000 | Good: 李太太静静弹着玻璃珠。<br>Bad: 李太太弹着静静玻璃珠。 |
| multiple edits: 对 -> 比较高兴; bad deletes 比较高兴 | adjective_transitive_dui | 11 | 0.9091 | 1.0000 | -0.0909 | 0.0000 | Good: 他们对张夫人的行为比较高兴。<br>Bad: 他们比较高兴张夫人的行为。 |
| 说 → 劝 | ya_insertion | 50 | 0.9600 | 0.8800 | +0.0800 | 0.0000 | Good: 张三说呀，消防员去清蒸鸭了。<br>Bad: 张三劝呀，消防员去清蒸鸭了。 |
| multiple edits: bad inserts 把我; bad deletes 把我 | verb_phrase_left_negation | 13 | 0.9231 | 1.0000 | -0.0769 | 0.0000 | Good: 你们没有把我夸奖。<br>Bad: 你们把我没有夸奖。 |
| multiple edits: bad inserts 拿手套; bad deletes 拿手套 | left_adverbial_e | 28 | 0.7143 | 0.6429 | +0.0714 | 0.0000 | Good: 你慢吞吞地拿手套给她。<br>Bad: 你拿手套慢吞吞地给她。 |
| multiple edits: bad inserts 喝过; bad deletes 喝过 | left_adverbial_d | 15 | 0.8667 | 0.9333 | -0.0667 | 0.0000 | Good: 你鼓起勇气喝过可乐。<br>Bad: 你喝过鼓起勇气可乐。 |
| multiple edits: bad inserts 喝; bad deletes 喝了 | preposition_insertion | 72 | 1.0000 | 0.9444 | +0.0556 | 0.0000 | Good: 她们给王先生的母亲喝了好几十瓶白酒。<br>Bad: 她们喝给王先生的母亲好几十瓶白酒。 |
| 觉得 → 告诉 | ya_insertion | 38 | 0.0789 | 0.1316 | -0.0526 | 0.0000 | Good: 我觉得呀，舞者去屠宰牛了。<br>Bad: 我告诉呀，舞者去屠宰牛了。 |
| multiple edits: bad inserts 拿裙子; bad deletes 拿裙子 | left_adverbial_e | 21 | 0.8571 | 0.8095 | +0.0476 | 0.0000 | Good: 你们很快地拿裙子给王大娘。<br>Bad: 你们拿裙子很快地给王大娘。 |
| multiple edits: 给 -> 过; bad deletes 鱼; 了 -> 鱼 | right_yijing_a | 25 | 1.0000 | 0.9600 | +0.0400 | 0.0000 | Good: 王小姐借给他们鱼已经五次了。<br>Bad: 王小姐借过他们已经五次鱼。 |
| bad deletes 在 | preposition_deletion | 300 | 0.9367 | 0.9000 | +0.0367 | 0.0000 | Good: 周大妈在火山上吃过好几串香蕉。<br>Bad: 周大妈火山上吃过好几串香蕉。 |
| 认为 → 告知 | ya_insertion | 39 | 0.7949 | 0.7692 | +0.0256 | 0.0000 | Good: 王姨认为呀，姐妹去喝橙汁了。<br>Bad: 王姨告知呀，姐妹去喝橙汁了。 |
| 认为 → 告诉 | ya_insertion | 54 | 0.0926 | 0.1111 | -0.0185 | 0.0000 | Good: 张夫人认为呀，打工人去制作手账了。<br>Bad: 张夫人告诉呀，打工人去制作手账了。 |
| multiple edits: bad inserts 吃; bad deletes 吃了 | preposition_insertion | 97 | 0.9897 | 0.9794 | +0.0103 | 0.0000 | Good: 她给那六位打工人吃了七个花卷。<br>Bad: 她吃给那六位打工人七个花卷。 |
| 以为 → 告诉 | ya_insertion | 37 | 0.0811 | 0.0811 | +0.0000 | 0.0000 | Good: 我们以为呀，领导去煮鸡了。<br>Bad: 我们告诉呀，领导去煮鸡了。 |
| multiple edits: bad inserts 努力; bad deletes 努力 | left_adverbial_negation | 20 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六位母亲没有努力拉大提琴。<br>Bad: 这六位母亲努力没有拉大提琴。 |
| multiple edits: bad inserts 轻声; bad deletes 轻声 | left_adverbial_negation | 20 | 0.9500 | 0.9500 | +0.0000 | 0.0000 | Good: 她没有轻声煮鸭。<br>Bad: 她轻声没有煮鸭。 |
| multiple edits: bad inserts 把他; bad deletes 把他 | verb_phrase_left_negation | 18 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没有把他嘉奖。<br>Bad: 你们把他没有嘉奖。 |
| multiple edits: bad inserts 拿裤子; bad deletes 拿裤子 | left_adverbial_e | 18 | 0.9444 | 0.9444 | +0.0000 | 0.0000 | Good: 她轻飘飘地拿裤子给我。<br>Bad: 她拿裤子轻飘飘地给我。 |
| multiple edits: bad inserts 唱过; bad deletes 唱过 | left_adverbial_d | 15 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 张夫人小声唱过京剧。<br>Bad: 张夫人唱过小声京剧。 |
| multiple edits: bad inserts 把你; bad deletes 把你 | verb_phrase_left_negation | 15 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们没有把你鼓励。<br>Bad: 我们把你没有鼓励。 |
| multiple edits: bad inserts 偷偷; bad deletes 偷偷 | left_adverbial_negation | 13 | 0.7692 | 0.7692 | +0.0000 | 0.0000 | Good: 你们没有偷偷包扎鼻子。<br>Bad: 你们偷偷没有包扎鼻子。 |
| multiple edits: bad inserts 创作; bad deletes 创作了 | preposition_insertion | 13 | 0.9231 | 0.9231 | +0.0000 | 0.0000 | Good: 张婶给那九个姐姐创作了几部小说。<br>Bad: 张婶创作给那九个姐姐几部小说。 |
| multiple edits: 对 -> 非常失望; bad deletes 非常失望 | adjective_transitive_dui | 13 | 0.9231 | 0.9231 | +0.0000 | 0.0000 | Good: 他们对李四的表现非常失望。<br>Bad: 他们非常失望李四的表现。 |
| multiple edits: bad inserts 专心; bad deletes 专心 | left_adverbial_negation | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太没有专心炖鱼。<br>Bad: 李太太专心没有炖鱼。 |
| multiple edits: bad inserts 检查; bad deletes 检查了 | preposition_insertion | 12 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈给那位消费者检查了九个眼睛。<br>Bad: 郑大妈检查给那位消费者九个眼睛。 |
| multiple edits: bad inserts 看着; bad deletes 看着 | left_adverbial_b | 11 | 0.7273 | 0.7273 | +0.0000 | 0.0000 | Good: 另外四位司机急速看着小说。<br>Bad: 另外四位司机看着急速小说。 |
| multiple edits: 对 -> 非常困惑; bad deletes 非常困惑 | adjective_transitive_dui | 11 | 0.9091 | 0.9091 | +0.0000 | 0.0000 | Good: 她对王五的所作所为非常困惑。<br>Bad: 她非常困惑王五的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 大象; 了 -> 大象 | right_yijing_a | 11 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 冯大哥寄给我大象已经七次了。<br>Bad: 冯大哥寄过我已经七次大象。 |
| multiple edits: 给 -> 过; bad deletes 玻璃珠; 了 -> 玻璃珠 | right_yijing_a | 10 | 0.9000 | 0.9000 | +0.0000 | 0.0000 | Good: 王大娘寄给我玻璃珠已经七次了。<br>Bad: 王大娘寄过我已经七次玻璃珠。 |
| multiple edits: bad inserts 看; bad deletes 看了 | preposition_insertion | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太给这个吉他手看了五本书。<br>Bad: 何太太看给这个吉他手五本书。 |
| multiple edits: 对 -> 很开心; bad deletes 很开心 | adjective_transitive_dui | 9 | 0.8889 | 0.8889 | +0.0000 | 0.0000 | Good: 她们对冯大哥的所作所为很开心。<br>Bad: 她们很开心冯大哥的所作所为。 |
| multiple edits: 对 -> 很苦恼; bad deletes 很苦恼 | adjective_transitive_dui | 9 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们对王姨的所作所为很苦恼。<br>Bad: 她们很苦恼王姨的所作所为。 |
| multiple edits: bad deletes 大象; bad inserts 大象 | right_yijing_b | 8 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 我们寄给王小姐大象已经非常多次了。<br>Bad: 我们寄给王小姐已经非常多次大象了。 |
| multiple edits: bad deletes 裤子; bad inserts 裤子 | right_yijing_b | 8 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 我寄给我们裤子已经三次了。<br>Bad: 我寄给我们已经三次裤子了。 |
| multiple edits: bad inserts 把她; bad deletes 把她 | verb_phrase_left_negation | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没有把她欺负。<br>Bad: 你们把她没有欺负。 |
| multiple edits: bad inserts 拍摄过; bad deletes 拍摄过 | left_adverbial_d | 8 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 冯大哥鼓起勇气拍摄过电影。<br>Bad: 冯大哥拍摄过鼓起勇气电影。 |
| multiple edits: bad inserts 拿椅子; bad deletes 拿椅子 | left_adverbial_e | 8 | 0.6250 | 0.6250 | +0.0000 | 0.0000 | Good: 你缓缓地拿椅子给你们。<br>Bad: 你拿椅子缓缓地给你们。 |
| multiple edits: 对 -> 很沉默; bad deletes 很沉默 | adjective_transitive_dui | 8 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他对徐小姐的行为很沉默。<br>Bad: 他很沉默徐小姐的行为。 |
| multiple edits: 对 -> 有点快乐; bad deletes 有点快乐 | adjective_transitive_dui | 8 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 他们对张婶的所作所为有点快乐。<br>Bad: 他们有点快乐张婶的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 裤子; 了 -> 裤子 | right_yijing_a | 8 | 0.8750 | 0.8750 | +0.0000 | 0.0000 | Good: 他们寄给杨大哥裤子已经八次了。<br>Bad: 他们寄过杨大哥已经八次裤子。 |
| multiple edits: bad deletes 努力; bad inserts 努力 | left_adverbial_d | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这六位服务员努力创作过小说。<br>Bad: 这六位服务员创作过努力小说。 |
| multiple edits: bad inserts 吃过; bad deletes 吃过 | left_adverbial_d | 7 | 0.8571 | 0.8571 | +0.0000 | 0.0000 | Good: 杨大哥红着脸吃过鸭。<br>Bad: 杨大哥吃过红着脸鸭。 |
| multiple edits: 对 -> 比较失望; bad deletes 比较失望 | adjective_transitive_dui | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她对张三的行为比较失望。<br>Bad: 她比较失望张三的行为。 |
| multiple edits: 对 -> 非常快乐; bad deletes 非常快乐 | adjective_transitive_dui | 7 | 0.7143 | 0.7143 | +0.0000 | 0.0000 | Good: 他对小明的所作所为非常快乐。<br>Bad: 他非常快乐小明的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 电视机; 了 -> 电视机 | right_yijing_a | 7 | 0.8571 | 0.8571 | +0.0000 | 0.0000 | Good: 陈大姐递给她们电视机已经十几次了。<br>Bad: 陈大姐递过她们已经十几次电视机。 |
| multiple edits: 给 -> 过; bad deletes 糖果; 了 -> 糖果 | right_yijing_a | 7 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太寄给我们糖果已经好几百次了。<br>Bad: 李太太寄过我们已经好几百次糖果。 |
| multiple edits: bad deletes 很快地; bad inserts 很快地 | left_adverbial_e | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们很快地拿开瓶器给李四。<br>Bad: 她们拿开瓶器很快地给李四。 |
| multiple edits: bad deletes 电冰箱; bad inserts 电冰箱 | right_yijing_b | 6 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 刘先生寄给冯大哥电冰箱已经好几次了。<br>Bad: 刘先生寄给冯大哥已经好几次电冰箱了。 |
| multiple edits: bad inserts 们把她; bad deletes 把她们 | verb_phrase_left_negation | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们没有把她们提醒。<br>Bad: 你们把她们没有提醒。 |
| multiple edits: bad inserts 拿桌子; bad deletes 拿桌子 | left_adverbial_e | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她努力地拿桌子给张先生。<br>Bad: 她拿桌子努力地给张先生。 |
| multiple edits: bad inserts 拿热水器; bad deletes 拿热水器 | left_adverbial_e | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 你慢吞吞地拿热水器给周大妈。<br>Bad: 你拿热水器慢吞吞地给周大妈。 |
| multiple edits: bad inserts 检查着; bad deletes 检查着 | left_adverbial_b | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 这个上级嘟着嘴检查着腿。<br>Bad: 这个上级检查着嘟着嘴腿。 |
| multiple edits: bad inserts 演奏过; bad deletes 演奏过 | left_adverbial_d | 6 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 李太太闭着眼睛演奏过协奏曲。<br>Bad: 李太太演奏过闭着眼睛协奏曲。 |
| multiple edits: 对 -> 很失望; bad deletes 很失望 | adjective_transitive_dui | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们对小王的行为很失望。<br>Bad: 他们很失望小王的行为。 |
| multiple edits: 对 -> 有点冷静; bad deletes 有点冷静 | adjective_transitive_dui | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 他们对张婶的所作所为有点冷静。<br>Bad: 他们有点冷静张婶的所作所为。 |
| multiple edits: 对 -> 比较开心; bad deletes 比较开心 | adjective_transitive_dui | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 她们对胡大爷的所作所为比较开心。<br>Bad: 她们比较开心胡大爷的所作所为。 |
| multiple edits: 对 -> 比较悲伤; bad deletes 比较悲伤 | adjective_transitive_dui | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 他对王小姐的所作所为比较悲伤。<br>Bad: 他比较悲伤王小姐的所作所为。 |
| multiple edits: 对 -> 比较愤怒; bad deletes 比较愤怒 | adjective_transitive_dui | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们对吴太太的行为比较愤怒。<br>Bad: 他们比较愤怒吴太太的行为。 |
| multiple edits: 对 -> 非常愤怒; bad deletes 非常愤怒 | adjective_transitive_dui | 6 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们对王先生的所作所为非常愤怒。<br>Bad: 她们非常愤怒王先生的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 糖; 了 -> 糖 | right_yijing_a | 6 | 0.8333 | 0.8333 | +0.0000 | 0.0000 | Good: 你们寄给赵大爷糖已经好几十次了。<br>Bad: 你们寄过赵大爷已经好几十次糖。 |
| multiple edits: bad deletes 专心地; bad inserts 专心地 | left_adverbial_e | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生专心地拿饮料瓶给胡大爷。<br>Bad: 张先生拿饮料瓶专心地给胡大爷。 |
| multiple edits: bad deletes 悄悄地; bad inserts 悄悄地 | left_adverbial_e | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈悄悄地拿玻璃珠给王五。<br>Bad: 周大妈拿玻璃珠悄悄地给王五。 |
| multiple edits: bad deletes 慢慢地; bad inserts 慢慢地 | left_adverbial_e | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们慢慢地拿饮料瓶给他们。<br>Bad: 我们拿饮料瓶慢慢地给他们。 |
| multiple edits: bad deletes 缓缓地; bad inserts 缓缓地 | left_adverbial_e | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她缓缓地拿玻璃珠给你们。<br>Bad: 她拿玻璃珠缓缓地给你们。 |
| multiple edits: bad deletes 蛋糕; bad inserts 蛋糕 | right_yijing_b | 5 | 0.6000 | 0.6000 | +0.0000 | 0.0000 | Good: 她递给他们蛋糕已经许多次了。<br>Bad: 她递给他们已经许多次蛋糕了。 |
| multiple edits: bad inserts 们把你; bad deletes 把你们 | verb_phrase_left_negation | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们没有把你们称赞。<br>Bad: 他们把你们没有称赞。 |
| multiple edits: bad inserts 清洗; bad deletes 清洗了 | preposition_insertion | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我给这位记者清洗了一个杯子。<br>Bad: 我清洗给这位记者一个杯子。 |
| multiple edits: bad inserts 清蒸着; bad deletes 清蒸着 | left_adverbial_b | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们低着头清蒸着鸭。<br>Bad: 她们清蒸着低着头鸭。 |
| multiple edits: bad inserts 观看; bad deletes 观看了 | preposition_insertion | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们给你们的妹妹观看了非常多部电影。<br>Bad: 我们观看给你们的妹妹非常多部电影。 |
| multiple edits: 给 -> 过; bad deletes 教材; 了 -> 教材 | right_yijing_a | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李四寄给他们教材已经非常多次了。<br>Bad: 李四寄过他们已经非常多次教材。 |
| multiple edits: 给 -> 过; bad deletes 衣服; 了 -> 衣服 | right_yijing_a | 5 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你们递给周大妈衣服已经好几百次了。<br>Bad: 你们递过周大妈已经好几百次衣服。 |
| multiple edits: bad deletes 专心; bad inserts 专心 | left_adverbial_d | 4 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 她们专心清蒸过鱼。<br>Bad: 她们清蒸过专心鱼。 |
| multiple edits: bad deletes 努力; bad inserts 努力 | left_adverbial_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生努力演奏着奏鸣曲。<br>Bad: 张先生演奏着努力奏鸣曲。 |
| multiple edits: bad deletes 小心地; bad inserts 小心地 | left_adverbial_e | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他小心地拿热水器给张夫人。<br>Bad: 他拿热水器小心地给张夫人。 |
| multiple edits: bad deletes 小猫; bad inserts 小猫 | right_yijing_b | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们递给你小猫已经八次了。<br>Bad: 她们递给你已经八次小猫了。 |
| multiple edits: bad deletes 杯子; bad inserts 杯子 | right_yijing_b | 4 | 0.7500 | 0.7500 | +0.0000 | 0.0000 | Good: 王姨递给她们杯子已经九次了。<br>Bad: 王姨递给她们已经九次杯子了。 |
| multiple edits: bad deletes 玻璃珠; bad inserts 玻璃珠 | right_yijing_b | 4 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 小明递给李太太玻璃珠已经十几次了。<br>Bad: 小明递给李太太已经十几次玻璃珠了。 |
| multiple edits: bad inserts 拿电冰箱; bad deletes 拿电冰箱 | left_adverbial_e | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太不遗余力地拿电冰箱给王姨。<br>Bad: 何太太拿电冰箱不遗余力地给王姨。 |
| multiple edits: bad inserts 清蒸; bad deletes 清蒸了 | preposition_insertion | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们给这个工人清蒸了好几条鱼。<br>Bad: 她们清蒸给这个工人好几条鱼。 |
| multiple edits: bad inserts 爆炒; bad deletes 爆炒了 | preposition_insertion | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你给周大妈的父亲爆炒了非常多只鸡。<br>Bad: 你爆炒给周大妈的父亲非常多只鸡。 |
| multiple edits: bad inserts 领养; bad deletes 领养了 | preposition_insertion | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们给那个弟弟领养了好几十只小猫。<br>Bad: 我们领养给那个弟弟好几十只小猫。 |
| multiple edits: 对 -> 有点失望; bad deletes 有点失望 | adjective_transitive_dui | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他对王姨的表现有点失望。<br>Bad: 他有点失望王姨的表现。 |
| multiple edits: 对 -> 比较沉默; bad deletes 比较沉默 | adjective_transitive_dui | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她对李先生的表现比较沉默。<br>Bad: 她比较沉默李先生的表现。 |
| multiple edits: 对 -> 非常伤心; bad deletes 非常伤心 | adjective_transitive_dui | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他对小王的所作所为非常伤心。<br>Bad: 他非常伤心小王的所作所为。 |
| multiple edits: 给 -> 过; bad deletes 书; 了 -> 书 | right_yijing_a | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王寄给郑大妈书已经好几次了。<br>Bad: 小王寄过郑大妈已经好几次书。 |
| multiple edits: 给 -> 过; bad deletes 啤酒; 了 -> 啤酒 | right_yijing_a | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐递给她们啤酒已经十次了。<br>Bad: 王小姐递过她们已经十次啤酒。 |
| multiple edits: 给 -> 过; bad deletes 热水器; 了 -> 热水器 | right_yijing_a | 4 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他借给我热水器已经九次了。<br>Bad: 他借过我已经九次热水器。 |
| multiple edits: bad deletes 轻轻地; bad inserts 轻轻地 | left_adverbial_e | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 何太太轻轻地拿充电器给宋女士。<br>Bad: 何太太拿充电器轻轻地给宋女士。 |
| multiple edits: bad inserts 们把我; bad deletes 把我们 | verb_phrase_left_negation | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们没有把我们批评。<br>Bad: 她们把我们没有批评。 |
| multiple edits: bad inserts 屠宰; bad deletes 屠宰了 | preposition_insertion | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王给你们的朋友屠宰了四头牛。<br>Bad: 小王屠宰给你们的朋友四头牛。 |
| multiple edits: bad inserts 烧; bad deletes 烧了 | preposition_insertion | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 胡大爷给她的上级烧了七条鱼。<br>Bad: 胡大爷烧给她的上级七条鱼。 |
| multiple edits: 对 -> 很伤心; bad deletes 很伤心 | adjective_transitive_dui | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 她们对张婶的表现很伤心。<br>Bad: 她们很伤心张婶的表现。 |
| multiple edits: 对 -> 很困惑; bad deletes 很困惑 | adjective_transitive_dui | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们对张夫人的表现很困惑。<br>Bad: 她们很困惑张夫人的表现。 |
| multiple edits: 给 -> 过; bad deletes 小猫; 了 -> 小猫 | right_yijing_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 吴太太借给她小猫已经非常多次了。<br>Bad: 吴太太借过她已经非常多次小猫。 |
| multiple edits: 给 -> 过; bad deletes 手账; 了 -> 手账 | right_yijing_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们借给宋女士手账已经八次了。<br>Bad: 她们借过宋女士已经八次手账。 |
| multiple edits: 给 -> 过; bad deletes 方便面; 了 -> 方便面 | right_yijing_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他送给小王方便面已经好几十次了。<br>Bad: 他送过小王已经好几十次方便面。 |
| multiple edits: 给 -> 过; bad deletes 桌子; 了 -> 桌子 | right_yijing_a | 3 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太寄给张先生桌子已经好几百次了。<br>Bad: 李太太寄过张先生已经好几百次桌子。 |
| multiple edits: 给 -> 过; bad deletes 电冰箱; 了 -> 电冰箱 | right_yijing_a | 3 | 0.6667 | 0.6667 | +0.0000 | 0.0000 | Good: 你送给何太太电冰箱已经八次了。<br>Bad: 你送过何太太已经八次电冰箱。 |
| multiple edits: bad deletes 小说; bad inserts 小说 | right_yijing_b | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我递给你小说已经九次了。<br>Bad: 我递给你已经九次小说了。 |
| multiple edits: bad deletes 桌子; bad inserts 桌子 | right_yijing_b | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你借给他桌子已经五次了。<br>Bad: 你借给他已经五次桌子了。 |
| multiple edits: bad deletes 红茶; bad inserts 红茶 | right_yijing_b | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 他们寄给你们红茶已经几次了。<br>Bad: 他们寄给你们已经几次红茶了。 |
| multiple edits: bad deletes 蛋炒饭; bad inserts 蛋炒饭 | right_yijing_b | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 她们借给王姨蛋炒饭已经四次了。<br>Bad: 她们借给王姨已经四次蛋炒饭了。 |
| multiple edits: bad deletes 轻声地; bad inserts 轻声地 | left_adverbial_e | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我轻声地拿录像带给你。<br>Bad: 我拿录像带轻声地给你。 |
| multiple edits: 对 -> 很冷静; bad deletes 很冷静 | adjective_transitive_dui | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们对小王的表现很冷静。<br>Bad: 她们很冷静小王的表现。 |
| multiple edits: 对 -> 很愤怒; bad deletes 很愤怒 | adjective_transitive_dui | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她对张夫人的表现很愤怒。<br>Bad: 她很愤怒张夫人的表现。 |
| multiple edits: 对 -> 有点开心; bad deletes 有点开心 | adjective_transitive_dui | 2 | 0.5000 | 0.5000 | +0.0000 | 0.0000 | Good: 他对李太太的行为有点开心。<br>Bad: 他有点开心李太太的行为。 |
| multiple edits: 对 -> 非常难过; bad deletes 非常难过 | adjective_transitive_dui | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们对张三的行为非常难过。<br>Bad: 她们非常难过张三的行为。 |
| multiple edits: 给 -> 过; bad deletes 作业; 了 -> 作业 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人递给周大妈作业已经七次了。<br>Bad: 张夫人递过周大妈已经七次作业。 |
| multiple edits: 给 -> 过; bad deletes 双簧; 了 -> 双簧 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈借给我双簧已经许多次了。<br>Bad: 周大妈借过我已经许多次双簧。 |
| multiple edits: 给 -> 过; bad deletes 可乐; 了 -> 可乐 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们寄给他们可乐已经十次了。<br>Bad: 她们寄过他们已经十次可乐。 |
| multiple edits: 给 -> 过; bad deletes 巧克力; 了 -> 巧克力 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我寄给李四巧克力已经几次了。<br>Bad: 我寄过李四已经几次巧克力。 |
| multiple edits: 给 -> 过; bad deletes 椅子; 了 -> 椅子 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王小姐送给吴太太椅子已经好几百次了。<br>Bad: 王小姐送过吴太太已经好几百次椅子。 |
| multiple edits: 给 -> 过; bad deletes 橙汁; 了 -> 橙汁 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们送给她橙汁已经三次了。<br>Bad: 我们送过她已经三次橙汁。 |
| multiple edits: 给 -> 过; bad deletes 红酒; 了 -> 红酒 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生寄给李四红酒已经六次了。<br>Bad: 刘先生寄过李四已经六次红酒。 |
| multiple edits: 给 -> 过; bad deletes 被子; 了 -> 被子 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她借给她们被子已经七次了。<br>Bad: 她借过她们已经七次被子。 |
| multiple edits: 给 -> 过; bad deletes 面包; 了 -> 面包 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李太太递给赵大爷面包已经四次了。<br>Bad: 李太太递过赵大爷已经四次面包。 |
| multiple edits: 给 -> 过; bad deletes 鱼丸; 了 -> 鱼丸 | right_yijing_a | 2 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他们寄给宋女士鱼丸已经许多次了。<br>Bad: 他们寄过宋女士已经许多次鱼丸。 |
| multiple edits: bad deletes 双簧; bad inserts 双簧 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王姨递给我双簧已经十次了。<br>Bad: 王姨递给我已经十次双簧了。 |
| multiple edits: bad deletes 古筝; bad inserts 古筝 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们寄给我古筝已经六次了。<br>Bad: 我们寄给我已经六次古筝了。 |
| multiple edits: bad deletes 坚果; bad inserts 坚果 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 你寄给我们坚果已经十几次了。<br>Bad: 你寄给我们已经十几次坚果了。 |
| multiple edits: bad deletes 大声; bad inserts 声大 | left_adverbial_d | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我大声麻醉过大象。<br>Bad: 我麻醉过大声大象。 |
| multiple edits: bad deletes 大提琴; bad inserts 大提琴 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈递给陈大姐大提琴已经好几次了。<br>Bad: 郑大妈递给陈大姐已经好几次大提琴了。 |
| multiple edits: bad deletes 小声; bad inserts 声小 | left_adverbial_d | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 我小声领养过小狗。<br>Bad: 我领养过小声小狗。 |
| multiple edits: bad deletes 小提琴; bad inserts 小提琴 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小明递给杨大哥小提琴已经六次了。<br>Bad: 小明递给杨大哥已经六次小提琴了。 |
| multiple edits: bad deletes 巧克力; bad inserts 巧克力 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她借给徐小姐巧克力已经好几十次了。<br>Bad: 她借给徐小姐已经好几十次巧克力了。 |
| multiple edits: bad deletes 日记; bad inserts 日记 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她寄给冯大哥日记已经十几次了。<br>Bad: 她寄给冯大哥已经十几次日记了。 |
| multiple edits: bad deletes 漫画; bad inserts 漫画 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 周大妈寄给郑大妈漫画已经好几百次了。<br>Bad: 周大妈寄给郑大妈已经好几百次漫画了。 |
| multiple edits: bad deletes 牛奶; bad inserts 牛奶 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生递给他牛奶已经好几百次了。<br>Bad: 李先生递给他已经好几百次牛奶了。 |
| multiple edits: bad deletes 苹果; bad inserts 苹果 | right_yijing_b | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈寄给你们苹果已经十次了。<br>Bad: 郑大妈寄给你们已经十次苹果了。 |
| multiple edits: bad deletes 馒头; bad inserts 馒头 | right_yijing_b | 1 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | Good: 你们递给王小姐馒头已经九次了。<br>Bad: 你们递给王小姐已经九次馒头了。 |
| multiple edits: bad inserts 先生把刘; bad deletes 把刘先生 | verb_phrase_left_negation | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 李先生没有把刘先生称赞。<br>Bad: 李先生把刘先生没有称赞。 |
| multiple edits: bad inserts 打断; bad deletes 打断了 | preposition_insertion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我们给另外九位上级打断了四只脚。<br>Bad: 我们打断给另外九位上级四只脚。 |
| multiple edits: bad inserts 拍摄; bad deletes 拍摄了 | preposition_insertion | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人给我的女儿拍摄了七部电视剧。<br>Bad: 张夫人拍摄给我的女儿七部电视剧。 |
| multiple edits: 对 -> 比较冷静; bad deletes 比较冷静 | adjective_transitive_dui | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她对冯大哥的表现比较冷静。<br>Bad: 她比较冷静冯大哥的表现。 |
| multiple edits: 给 -> 过; bad deletes 咖啡; 了 -> 咖啡 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈寄给她咖啡已经好几十次了。<br>Bad: 郑大妈寄过她已经好几十次咖啡。 |
| multiple edits: 给 -> 过; bad deletes 小提琴; 了 -> 小提琴 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张夫人递给她小提琴已经好几次了。<br>Bad: 张夫人递过她已经好几次小提琴。 |
| multiple edits: 给 -> 过; bad deletes 录像带; 了 -> 录像带 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 我借给他录像带已经八次了。<br>Bad: 我借过他已经八次录像带。 |
| multiple edits: 给 -> 过; bad deletes 收音机; 了 -> 收音机 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 王先生递给宋女士收音机已经好几十次了。<br>Bad: 王先生递过宋女士已经好几十次收音机。 |
| multiple edits: 给 -> 过; bad deletes 日记; 了 -> 日记 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 张先生送给冯大哥日记已经好几次了。<br>Bad: 张先生送过冯大哥已经好几次日记。 |
| multiple edits: 给 -> 过; bad deletes 橘子; 了 -> 橘子 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 杨大哥借给徐小姐橘子已经四次了。<br>Bad: 杨大哥借过徐小姐已经四次橘子。 |
| multiple edits: 给 -> 过; bad deletes 漫画; 了 -> 漫画 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 刘先生递给你们漫画已经四次了。<br>Bad: 刘先生递过你们已经四次漫画。 |
| multiple edits: 给 -> 过; bad deletes 笛子; 了 -> 笛子 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他寄给张夫人笛子已经几次了。<br>Bad: 他寄过张夫人已经几次笛子。 |
| multiple edits: 给 -> 过; bad deletes 红茶; 了 -> 红茶 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 郑大妈借给我们红茶已经三次了。<br>Bad: 郑大妈借过我们已经三次红茶。 |
| multiple edits: 给 -> 过; bad deletes 葡萄汁; 了 -> 葡萄汁 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 宋女士送给我葡萄汁已经五次了。<br>Bad: 宋女士送过我已经五次葡萄汁。 |
| multiple edits: 给 -> 过; bad deletes 蛋炒饭; 了 -> 蛋炒饭 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 他递给杨大哥蛋炒饭已经许多次了。<br>Bad: 他递过杨大哥已经许多次蛋炒饭。 |
| multiple edits: 给 -> 过; bad deletes 钢琴; 了 -> 钢琴 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 她们送给你钢琴已经八次了。<br>Bad: 她们送过你已经八次钢琴。 |
| multiple edits: 给 -> 过; bad deletes 香蕉; 了 -> 香蕉 | right_yijing_a | 1 | 1.0000 | 1.0000 | +0.0000 | 0.0000 | Good: 小王递给何太太香蕉已经七次了。<br>Bad: 小王递过何太太已经七次香蕉。 |

