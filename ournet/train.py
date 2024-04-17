import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from ournet import crnn as crnn, src as utils, src as dataset

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_list', default='data/train.txt', type=str, help='path to train dataset list file')  # 训练集txt
parser.add_argument('--eval_list', type=str, default='data/test.txt', help='path to evalation dataset list file')  # 测试集txt
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading num_workers')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')  # 批大小
parser.add_argument('--img_height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--img_width', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--num_epochs', type=int, default=20, help='number of epochs to train for')  # 训练轮数
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')  # lr
parser.add_argument('--model_path', type=str, default="weights/model_11.pth",
                    help="path to model (to continue training)")  # 继续训练的模型路径
parser.add_argument('--use_existing', type=bool, default=True, help="use existing model (to continue training)")  # 是否使用已有模型
parser.add_argument('--model', default='weights/', help='Where to store samples and models')  # 模型保存路径
parser.add_argument('--random_sample', default=True, action='store_true',
                    help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.0, help='where to use teach forcing')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
cfg = parser.parse_args()
# print(cfg)

# load alphabet
with open('D:/ProgramCode/craft_rcnn_forch-master/craft_rcnn_forch/data/new_dict.txt', encoding="UTF-8") as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + EOS_TOKEN + BLANK_TOKEN
num_classes = len(alphabet) + 2


def train(model, criterion, train_loader, max_eval_iter, test_loader, teach_forcing_prob=1):
    # optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

    # loss averager
    loss_avg = utils.Averager()

    for epoch in range(cfg.num_epochs):
        model.train()
        print(f"epoch: {epoch}")
        train_iter = iter(train_loader)

        for i in range(len(train_loader)):
            cpu_images, cpu_texts = next(train_iter)

            target_variable, text_length = converter.encode(cpu_texts)
            # print(target_variable.shape) [22, 32]
            image = cpu_images.to("cuda")

            output = model(image)
            # print(output.shape) # [69, 32, 5992]
            input_length_with_blank = output.shape[0]
            actual_batch_size = target_variable.shape[1]

            # [predictseq_len, b, classnum], [b, longest_targetseq_len], [b](content is predictseq_len),
            # [b](content is actual_targetseq_len)
            loss = criterion(output, target_variable.transpose(0, 1),
                              torch.tensor([input_length_with_blank] * actual_batch_size), text_length)

            model.zero_grad()
            loss.backward()
            model_optimizer.step()
            loss_avg.add(loss)

            if i % 10 == 0:
                print('\r[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch, cfg.num_epochs, i, len(train_loader),
                                                                         loss_avg.val()), end="")
                loss_avg.reset()
        print()
        # evaluate per epoch after training
        evaluate(model, criterion, test_loader, max_eval_iter=max_eval_iter)

        # save checkpoint
        torch.save(model.state_dict(), '{0}/model_{1}.pth'.format(cfg.model, epoch))


def evaluate(model, criterion, data_loader, max_eval_iter=100):
    with torch.no_grad():
        model.eval()
        val_iter = iter(data_loader)

        n_correct = 0
        n_total = 0
        loss_avg = utils.Averager()

        for i in range(min(len(data_loader), max_eval_iter)):
            cpu_images, cpu_texts = next(val_iter)

            image = cpu_images.to("cuda")
            target_variable, text_length = converter.encode(cpu_texts)
            n_total += len(cpu_texts)

            output = model(image)  # [length, batch, classes]
            input_length_with_blank = output.shape[0]
            actual_batch_size = target_variable.shape[1]

            loss = criterion(output, target_variable.transpose(0, 1),
                             torch.tensor([input_length_with_blank] * actual_batch_size), text_length)
            topv, topi = output.detach().topk(1)  # [length, batch, 1]
            ni = topi.squeeze(2).transpose(0, 1)  # [batch, length]
            decoded_words = list(map(converter.decodeList, ni))
            loss_avg.add(loss)

            real_decoded = []
            for text in cpu_texts:
                real = []
                for word in text:
                    real.append(word) if word in alphabet else real.append("?")
                real_decoded.append("".join(real))

            for pred, target in zip(decoded_words, real_decoded):
                if pred == target:
                    n_correct += 1

            if i % 10 == 0:
                print('pred: {}, gt: {}, real:{}'.format(''.join(decoded_words[0]), cpu_texts[0], real_decoded[0]))

        accuracy = n_correct / float(n_total)
        print('Test loss: {}, accuracy: {}'.format(loss_avg.val(), accuracy))
        with open("records.txt", "a") as f:
            f.write('Test loss: {}, accuracy: {}\n'.format(loss_avg.val(), accuracy))
        loss_avg.reset()


def main():
    if not os.path.exists(cfg.model):
        os.makedirs(cfg.model)

    # create train dataset
    train_dataset = dataset.TextLineDataset(text_line_file=cfg.train_list, transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=sampler, num_workers=int(cfg.num_workers),
        collate_fn=dataset.AlignCollate(img_height=cfg.img_height, img_width=cfg.img_width))

    # create test dataset
    test_dataset = dataset.TextLineDataset(text_line_file=cfg.eval_list,
                                           transform=dataset.ResizeNormalize(img_width=cfg.img_width,
                                                                             img_height=cfg.img_height, strict=False))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1,
                                              num_workers=int(cfg.num_workers))

    # create model
    model = crnn.CRNN(3, cfg.img_height, cfg.img_width, num_classes)
    # print(model)
    model.apply(utils.weights_init)
    if cfg.use_existing:
        print('loading pretrained encoder model from %s' % cfg.model_path)
        model.load_state_dict(torch.load(cfg.model_path))

    criterion = torch.nn.CTCLoss(blank=1)

    assert torch.cuda.is_available(), "Please run \'train.py\' script on nvidia cuda devices."
    model.cuda()
    criterion = criterion.cuda()

    # train crnn
    train(model, criterion, train_loader, 300, test_loader, teach_forcing_prob=cfg.teaching_forcing_prob)

    # # do evaluation after training
    # evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=800)


if __name__ == "__main__":
    main()
