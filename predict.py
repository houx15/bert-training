import torch

import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import sentence_cleaner


class OpinionPredict(object):
    r"""
    Make predictions based on trained models.

    Args:
        task_type (`str`):
            Current task type, 'regression' or 'binary'.
        model_path (`str`):
            Model name or path, e.g. '../model-bert/gun-regression'. The model will be used for tokenization and prediction.
        src_type (`str`, *optional*, defaults to `tweet`):
            Data source. 'tweet' or 'weibo
    """

    def __init__(
        self, task_type: str, model_path: str, src_type: str = "weibo"
    ) -> None:
        assert task_type in ["regression", "binary"]
        self.task_type = task_type
        self.src_type = src_type

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=False)
        self.model = self.model_init(model_path)

    def model_init(self, model_path: str):
        print(">>>>>>>>>initializing model:  {}".format(model_path))

        num_labels_map = {
            "binary": 2,
            "regression": 1,
        }
        torch.cuda.empty_cache()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels_map[self.task_type],
            output_attentions=True,
            output_hidden_states=False,
        )
        if torch.cuda.is_available():
            model.cuda()
        self.model = model
        return self.model

    def predict(
        self,
        texts: list,
        max_length: int = None,
        padding: str = "max_length",
        batch: int = 512,
        verbose=print,
    ) -> np.array:
        """
        :param texts: list (or numpy.array, pandas.Series, torch.tensor) of strings.
        :param padding: str. 'longest', 'max_length' (default), 'do_not_pad'.
        :param max_length, batch: int. max length of tweet and number of tweets proceeded in a batch. limited by GPU memory.
            max_length=128 is sufficient for most tweets, and 512 tweeets per batch are recommended for 128-letter tweets on a typical Tesla GPU with 16GB memory.
            :param verbose: function that accepts string input. used to display message.
            e.g., verbose = print (default, display on screen)
            e.g., verbose = logger.info (write to a log file)
            e.g., verbose = lambda message: logger.info(f'filename: {message}') # write to a log file with a header
        :return: 1-dim numpy array
        :example:
            >>> from predict import OpinionPredict
            >>> OpinionPredict(task_type='regression', src_type='tweet', model_path='../model-bert/2022-06-24-yuxin-gun-regression'
            ).predict(texts=[
                "RT After the shooting, Jim and his wife Sarah dedicated their lives to preventing gun violence. They were lifelong Republicans and gun owners themselves. They realized that passing sensible gun laws isn't about politics; it's about saving lives. #GunReform ",
                'Did u hear the gunshot in the video, when someone is rushing you and u hear a gunshot, they could have a gun so u shoot them. Someone else fired a gun.',
                "It didn't though. You can literally 3d print a gun now anyways, no use in banning them. Also that's a one way ticket to all out civil war",
                'I repeat, the 2nd amendment is not on trial here. Kyle did not engage, others engaged him. You are allowed to eliminate as many threats as is necessary to preserve your own life. There is no limit after which the right to your own life becomes inferior to that of your attackers.'
            ])

            output: array([ 1.7457896 ,  1.0045699 ,  0.07550862, -0.76812345], dtype=float32)
            ground truth: [2, 1, 0, -1]

            >>> OpinionPredict(task_type='climate', src_type='weibo', model_path='../../model-bert/2022-11-04-yuxin-climate'
            ).predict(texts=[
                    # label 1
                    "#全球变暖##当前全球变暖程度两千年未遇#  http://f.us.sinaimg.cn/003buVDvlx07vIPXCxrq010412006FkO0E010.mp4?label=mp4_720p&template=720x1280.24.0&trans_finger=37e3fed30081d60f956dbe10b6ff7523&Expires=1564191000&ssig=CffYR9pMXq&KID=unistore,video",
                    "澳洲热的要死，美国东部冻得要死。中国忽冷忽热……极端气候越来越多#全球气候变化#  See extreme weather across the globe",
                    "天理不但规定着个人的因果报应，也规定了一个地区，一个朝代，一个国家甚至一个世界的因果报应。天灾人祸就是共业所造的恶之果报，温室效应就是一个最好的例子。一切都在天理的范畴之内。 【谁在操纵命运】 ",
                    # label 2
                    "地铁广播开始送圣诞祝福了 小温馨[心][心][心]。 门口为全球变暖论坛准备的冰因为最近降温越结越大[笑cry] 顺便吐槽下#全球变暖# 星球进化多么宏观的课题 人类不过寄居的🐜 是不是太把自己当回事了 如果人类不幸真的毁了星球 也不过是这个星球的宿命而已[吃瓜]#今日贴纸打卡#  ",
                    "发表了一篇转载博文《[转载]【斯诺登】“全球变暖是一个由中情局发明的骗局”》[转载]【斯诺登】“...",
                    "《全球变暖已不复存在，地球正处于变冷状态，科学家认为此事不简单》全球变暖已不复存在，地球正处于变冷状态，科学家认为此事不简单",
                    # label 3
                    "又有谁能想到 我在三十几度的高温下还在犯风湿 还要买膏药贴呢 这个天不开空调要热死 开了空调腿要痛死 大家冬天一定要穿秋裤 好好养生吧[跪了] ",
                    "今天老妈要回去了，伤感情绪戛然而生，回想起来这一个月里，自己连拖把都没有拾起过，真是惭愧！老妈因身体原因而不能吹空调，每天忍耐炎热高温还给我们做着各种家务，也不喊累，不喊热！来到陌生的环境，每天也就只能散散步看看电视搞搞卫生，消遣时间了。老妈在，每天才有可口的饭菜，才不会感到孤单 ",
                    "只有在刚入手新化妆品的那几天才会认真的抹脸，没过一个月就开始瞎涂了，还有和我一样的三分钟热度女孩吗[笑cry][笑cry][笑cry] "
                ])

            output:

                array([[ 2.55624795, -1.37531841, -2.05213404],
                        [ 2.51814318, -1.52832699, -1.71303737],
                        [ 0.04927956,  0.15148288, -0.77340394],
                        [ 2.56860423, -2.00646591, -1.2080487 ],
                        [-0.30993834,  0.95506799, -1.47718608],
                        [ 0.41168666,  0.30903456, -1.81370771],
                        [-1.64383245, -1.7373091 ,  3.69929457],
                        [-1.74833012, -1.41184819,  3.30613136],
                        [-1.17227781, -1.62402332,  3.16496921]])

            ground truth:
                [[1,0,0],
                    [1,0,0],
                    [1,0,0],
                    [0,1,0],
                    [0,1,0],
                    [0,1,0],
                    [0,0,1],
                    [0,0,1],
                    [0,0,1]]
        """

        verbose(
            f"predict(texts={len(texts)}, max_length={max_length}, padding={padding}, batch={batch})"
        )
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False  # freeze the finetuned model. save memory.
        torch.cuda.empty_cache()
        try:
            prediction = np.array([])
            verbose(f"initial prediction.shape={prediction.shape}")

            for i in range(0, len(texts), batch):
                # encode input texts
                encoding = self.tokenizer(
                    [
                        sentence_cleaner(self.src_type, single_text)
                        for single_text in texts[i : i + batch]
                    ],
                    add_special_tokens=True,
                    return_token_type_ids=True,
                    truncation=True,
                    padding=padding,
                    return_attention_mask=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                if torch.cuda.is_available():
                    # verbose(f'self.model.device={self.model.device}')  # self.model.device=cuda:0
                    for key in encoding.keys():
                        encoding[key] = encoding[key].cuda()
                        # verbose(f'encoding[{key}].device={encoding[key].device}. encoding[{key}].shape={encoding[key].shape}') # encoding[input_ids].device=cuda:0. encoding[input_ids].shape=torch.Size([4, 128])

                # calculate the encoded input with frozen model
                outputs = self.model(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                    token_type_ids=encoding["token_type_ids"],
                ).logits.detach()
                # verbose(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}') #  type(outputs)=<class 'torch.Tensor'>, outputs.device=cuda:0
                if torch.cuda.is_available():
                    outputs = (
                        outputs.cpu()
                    )  # copy the tensor to host memory before converting it to numpy. otherwise we will get an error "can't convert cuda:0 device type tensor to numpy"
                    # verbose(f'type(outputs)={type(outputs)}, outputs.device={outputs.device}')  # type(outputs)=<class 'torch.Tensor'>, outputs.device=cpu
                del encoding
                verbose(f"outputs.numpy().shape={outputs.numpy().shape}")
                # verbose(f'outputs.numpy()={outputs.numpy()}')

                # output
                if "regression" == self.task_type:
                    prediction = np.append(
                        prediction, outputs.numpy().flatten()
                    )  # a score for each input string
                elif "binary" == self.task_type:
                    prediction = np.append(
                        prediction, np.argmax(outputs.numpy(), axis=1)
                    )  # a float probability of label "1" for each input string
                else:
                    raise ValueError(f"Unknown self.task_type = {self.task_type}.")
                del outputs
                torch.cuda.empty_cache()
                verbose(f"prediction.shape={prediction.shape}")
                # verbose(f'prediction={prediction}')

            # output
            return prediction

        except RuntimeError as error:
            verbose(f"Running out of memory, retrying with a smaller batch.")
            raise RuntimeError(
                "Running out of GPU memory. Try limiting [max_length] and [batch]."
            ) from error

    # predict()
