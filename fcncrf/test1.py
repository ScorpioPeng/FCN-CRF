from __future__ import print_function
import tensorflow as tf
import numpy as np
import sklearn.metrics as me
import sklearn.exceptions
import scipy.misc as misc
import warnings

import TensorflowUtils1 as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
import fcn32_vgg 
import loss as LOSS
# from six.moves import xrange
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
# config.gpu_options.per_process_gpu_memory_fraction = 0.49# 占用GPU90%的显存 
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "20", "batch size for training")
tf.flags.DEFINE_float("weight_decay", "0.0005", "weight decay")
tf.flags.DEFINE_string("logs_dir", "log99/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_string('mode', "visualize", "Mode train/ test/ visualize")
# tf.flags.DEFINE_integer("val_batch_size", "40", "batch size for val")


MAX_ITERATION = int(210000 + 1)
num_classes = 2
IMAGE_SIZE = 256
val_batch_size = 40


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_hist(predictions, labels):
  num_class = predictions.shape[3]
  batch_size = predictions.shape[0]
  hist = np.zeros((num_class, num_class))
  for i in range(batch_size):
    hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
  return hist

def get_FM(predictions, labels):
    batch_size = predictions.shape[0]
    f1 = 0
    for i in range(batch_size):
        f1 += me.f1_score(labels[i].flatten(), predictions[i].argmax(2).flatten())
    return f1


def main(argv=None):

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name="annotation")
    FM_pl = tf.placeholder(tf.float32,[])
    total_acc_pl = tf.placeholder(tf.float32,[])
    acc_pl = tf.placeholder(tf.float32,[])
    iu_pl = tf.placeholder(tf.float32,[])
    fwavacc_pl = tf.placeholder(tf.float32,[])
    # is_traing = tf.placeholder('bool')

    vgg_fcn = fcn32_vgg.FCN32VGG()
    vgg_fcn.build(image, num_classes=num_classes, keep_probability=keep_probability, random_init_fc8=True)

    logits = vgg_fcn.upscore
    pred_annotation = vgg_fcn.pred_up
    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
    #                                                                       labels=tf.squeeze(annotation, squeeze_dims=[3]),
    #                                                                       name="entropy")))
    # loss_summary = tf.summary.scalar("entropy", loss)

    # trainable_var = tf.trainable_variables()
    # S_vars = [svar for svar in tf.trainable_variables() if 'weight' in svar.name]
    # l2 = tf.add_n([tf.nn.l2_loss(var) for var in S_vars])
    # # loss = loss + l2 * FLAGS.weight_decay
    # # train_op = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(loss + l2 * FLAGS.weight_decay)
    # train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss + l2 * FLAGS.weight_decay)
    # # train_op = train(loss, trainable_var)

    """ median-frequency re-weighting """
    # class_weights = np.array([
    #     0.5501, 
    #     5.4915 
    # ])
    # loss = tf.reduce_mean((tf.nn.weighted_cross_entropy_with_logits(logits=logits,
    #                                                                 targets=tf.one_hot(tf.squeeze(annotation, squeeze_dims=[3]), depth=num_classes),
    #                                                                 pos_weight=class_weights,
    #                                                                 name="entropy")))
    loss = LOSS.loss(logits, tf.one_hot(tf.squeeze(annotation, squeeze_dims=[3]), depth=num_classes))
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    t_loss = loss + regularization_loss

    loss_summary = tf.summary.scalar("entropy", loss)
    FM_summary = tf.summary.scalar('FM', FM_pl)
    acc_total_summary = tf.summary.scalar("total_acc", total_acc_pl)
    acc_summary = tf.summary.scalar("acc", acc_pl)
    iu_summary = tf.summary.scalar("iu", iu_pl)
    fwavacc_summary = tf.summary.scalar("fwavacc", fwavacc_pl)

    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
    #train_op = tf.train.MomentumOptimizer(FLAGS.learning_rate, 0.9).minimize(t_loss)
    # train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(t_loss)
    summary_op = tf.summary.merge_all()

    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session(config=config)
    saver = tf.train.Saver(max_to_keep=3)


    # create two summary writers to show training loss and validation loss in the same graph
    # need to create two folders 'train' and 'validation' inside FLAGS.logs_dir
    praph_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/graph', sess.graph)
    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train')
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in range(1, MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability:0.5}

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                train_writer.add_summary(summary_str, itr)

            if itr % 210 == 0:
                valid_iamges, valid_annotations = validation_dataset_reader.get_records()
                val_count = 0
                total_loss = 0
                hist = np.zeros((num_classes, num_classes))
                fm = 0
                for i in range(1, 21):
                    val_images = valid_iamges[val_count:val_count+val_batch_size]
                    val_annotations = valid_annotations[val_count:val_count+val_batch_size]
                    val_loss, val_pred_dense = sess.run([loss, logits],  feed_dict={image: val_images, annotation: val_annotations,
                                                       keep_probability:1.0})
                    total_loss = total_loss + val_loss
                    val_count = val_count + val_batch_size
                    hist += get_hist(val_pred_dense, val_annotations)
                    fm += get_FM(val_pred_dense, val_annotations)

                valid_loss = total_loss / 20
                FM = fm/(20*val_batch_size)
                acc_total = np.diag(hist).sum() / hist.sum()
                acc = np.diag(hist) / hist.sum(1)
                iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
                freq = hist.sum(1) / hist.sum()

                # summary_st = sess.run(summary_op,feed_dict=feed_dict)

                summary_sva = sess.run(loss_summary, feed_dict={loss:valid_loss})
                summary_FM = sess.run(FM_summary, feed_dict={FM_pl:FM})
                summary_acc_total = sess.run(acc_total_summary, feed_dict={total_acc_pl:acc_total})
                summary_acc = sess.run(acc_summary, feed_dict={acc_pl:np.nanmean(acc)})
                summary_iu = sess.run(iu_summary, feed_dict={iu_pl:np.nanmean(iu)})
                summary_fwavacc = sess.run(fwavacc_summary, feed_dict={fwavacc_pl:(freq[freq > 0] * iu[freq > 0]).sum()})
                print("Step: %d, Valid_loss:%g" % (itr, valid_loss))
                print(" >>> Step: %d, f1_score:%g" % (itr, FM))
                # overall accuracy
                print(" >>> Step: %d, overall accuracy:%g" % (itr, acc_total))
                print(" >>> Step: %d, mean accuracy:%g" % (itr, np.nanmean(acc)))
                print(" >>> Step: %d, mean IU:%g" % (itr, np.nanmean(iu)))
                print(" >>> Step: %d, fwavacc:%g" % (itr, (freq[freq > 0] * iu[freq > 0]).sum()))

                # validation_writer.add_summary(summary_st, step)
                validation_writer.add_summary(summary_sva, itr)
                validation_writer.add_summary(summary_FM, itr)
                validation_writer.add_summary(summary_acc_total, itr)
                validation_writer.add_summary(summary_acc, itr)
                validation_writer.add_summary(summary_iu, itr)
                validation_writer.add_summary(summary_fwavacc, itr)

                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)
                
                va_images, va_annotations = validation_dataset_reader.get_random_batch(20)

                pred = sess.run(pred_annotation, feed_dict={image: va_images, annotation: va_annotations,
                                                            keep_probability:1.0})
                va_annotations = np.squeeze(va_annotations, axis=3)
                # pred = np.squeeze(pred, axis=3)
                pred = pred * 255
                va_annotations = va_annotations * 255
                for it in range(20):
                    utils.save_image(va_images[it].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+it))
                    utils.save_image(va_annotations[it].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+it))
                    utils.save_image(pred[it].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+it))

    elif FLAGS.mode == "visualize":
        it=0
        valid_iamge, val_annotation = validation_dataset_reader.get_records()
        val_annotation = np.squeeze(val_annotation, axis=3)
        val_annotation=val_annotation*255
        for filename in valid_records:
            val_image = np.array(misc.imread(filename['image']))
            val_image = np.reshape(val_image, (1, 256, 256, 3))
            pred = sess.run(pred_annotation, feed_dict={image: val_image, 
                                                    keep_probability:1.0})
            pred=pred*255
            # pred = sess.run(pred_annotation, feed_dict={image: val_image, 
            #                                         keep_probability:1.0})
            utils.save_image(pred[0].astype(np.uint8), FLAGS.logs_dir+'pred01', name=os.path.splitext(filename['image'].split("/")[-1])[0])
            utils.save_image(val_annotation[it].astype(np.uint8), FLAGS.logs_dir+'gt01', name=os.path.splitext(filename['annotation'].split("/")[-1])[0])
            it=it+1

    # elif FLAGS.mode == "visualize":
    #     valid_images, valid_annotations = validation_dataset_reader.get_random_batch(20)
    #     pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
    #                                                 is_traing: False})
    #     valid_annotations = np.squeeze(valid_annotations, axis=3)
    #     pred = np.squeeze(pred, axis=3)

    #     for itr in range(20):
    #         utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
    #         utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
    #         utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
    #         print("Saved image: %d" % itr)


if __name__ == "__main__":
    tf.app.run()





