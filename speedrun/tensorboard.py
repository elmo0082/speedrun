from contextlib import contextmanager
try:
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import tensorboardX as tx
except ImportError:
    tx = None
    plt = None
    np = None


class TensorboardMixin(object):
    @property
    def logger(self):
        if tx is None:
            raise ModuleNotFoundError("TensorboardMixin requires tensorboardX. You can "
                                      "install it with `pip install tensorboardX`")
        # Build logger if it doesn't exist
        if not hasattr(self, '_logger'):
            # noinspection PyUnresolvedReferences,PyAttributeOutsideInit
            self._logger = tx.SummaryWriter(log_dir=self.log_directory)
            # noinspection PyUnresolvedReferences
            self._meta_config['exclude_attrs_from_save'].append('_logger')
        return self._logger

    @property
    def tagscope(self):
        if not hasattr(self, '_tagscope'):
            # noinspection PyAttributeOutsideInit
            self._tagscope = ''
        return self._tagscope

    @contextmanager
    def set_tagscope(self, name):
        try:
            self._tagscope = name
            yield
        finally:
            # noinspection PyAttributeOutsideInit
            self._tagscope = ''

    def get_full_tag(self, tag):
        if self.tagscope:
            return "{}/{}".format(self.tagscope, tag)
        else:
            return tag

    def log_scalar(self, tag, value, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        if self.log_scalars_now:
            self.logger.add_scalar(tag=self.get_full_tag(tag), scalar_value=value,
                                global_step=step)
        return self

    def log_image(self, tag, value, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        if self.log_images_now:
            self.logger.add_image(tag=self.get_full_tag(tag), img_tensor=value,
                                global_step=step)
        return self
    
    def log_persistence_diagram(self, tag, value, step=None):
        # noinspection PyUnresolvedReferences
        """
        This piece of code allows the logging of a persistence diagram as a 
        matplotlib figure and an associated image of the slice in question.
        The image contains a arrows at every pair of critical points pointing upward/downward,
        indicating whether the gradient there is positive or negative.
        Also the arrows have the same color as the corresponding point in the persistence diagram.

        :param tag: string
            Name tag for display in tensorboard.
        :param value: tuple
            A tuple containing a rgb-version of the probability map as well as a map of critical points adn gradients.
        """
        step = self.step if step is None else step
        if self.log_persistence_diagrams_now:
            rgb_in_pic, topo_grad = value

            # create figure object
            # also draw black line as orientation
            fig, ax = plt.subplots()
            x = np.linspace(0, 1.0, 5)
            ax.plot(x, x, 'k-')
            ax.set_xlabel('Birth time')
            ax.set_ylabel('Death time')

            # we have to add each point and the corresponding arrows of the persistence diagram separately
            for j in range(int(len(topo_grad) / 2)):
                i = 2*j
                birth_arrow = [topo_grad[i, 0], topo_grad[i, 1]]
                death_arrow = [topo_grad[i+1, 0], topo_grad[i+1, 1]]

                color = np.random.rand(3)
                rgb_in_pic = draw_arrow(rgb_in_pic, birth_arrow, color, topo_grad[i, 2])
                rgb_in_pic = draw_arrow(rgb_in_pic, death_arrow, color, topo_grad[i+1, 2])

                ax.scatter(topo_grad[i,3], topo_grad[i+1,3], color=(color[0], color[1], color[2]))

            self.logger.add_figure(tag=self.get_full_tag(tag), figure=fig, 
                                global_step=step)
            self.logger.add_image(tag=self.get_full_tag(tag + '_gradients'), img_tensor=torch.stack(rgb_in_pic, dim=0), 
                                global_step=step)                     
        return self

    def log_embedding(self, tag, tensor, images=None, metadata=None, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        if self.log_embeddings_now:
            self.logger.add_embedding(tag=self.get_full_tag(tag), mat=tensor,
                                    metadata=metadata, label_img=images, global_step=step)
        return self

    def log_histogram(self, tag, value, bins='tensorflow', step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        if self.log_histograms_now:
            self.logger.add_histogram(tag=self.get_full_tag(tag), values=value, bins=bins,
                                    global_step=step)
        return self

    def log_text(self, tag, text_string, step=None):
        # noinspection PyUnresolvedReferences
        step = self.step if step is None else step
        self.logger.add_text(tag=self.get_full_tag(tag), text_string=text_string,
                             global_step=step)

    def _log_x_now(self, x):
        # noinspection PyUnresolvedReferences
        frequency = self.get(f'trainer/tensorboard/log_{x}_every', 1)[0]
        if frequency is not None:
            # noinspection PyUnresolvedReferences
            return (self.step % frequency) == 0
        else:
            return False

    @property
    def log_scalars_now(self):
        return self._log_x_now('scalars')

    @property
    def log_images_now(self):
        return self._log_x_now('images')

    @property
    def log_figures_now(self):
        return self._log_x_now('figures')
    
    @property
    def log_persistence_diagrams_now(self):
        return self._log_x_now('persistence_diagrams')

    @property
    def log_embeddings_now(self):
        return self._log_x_now('embeddings')

    @property
    def log_histograms_now(self):
        return self._log_x_now('histograms')


def draw_arrow(image, position, color, grad):
    """
    Function that adds an arrow to a dataset at the position (i,j) with colormap (r,g,b)
    """
    # arrow that points upward
    sign = 1
    arrow_offsets = [[0, 0], [1,0], [2, 0], 
                     [3, 0], [4, 0], [5, 0], 
                     [1, 1], [1, -1], [2, 2], 
                     [2, -2], [2, 1], [2, -1], [6, 0]]

    # if gradient is negative, let the arrow point downward
    if grad < 0:
        sign = -1

    y = int(position[0])
    x = int(position[1])
    for offset in arrow_offsets:
        image[0][(x + sign*offset[0]) % image[0].shape[0]][(y + offset[1]) % image[0].shape[0]] = color[0]
        image[1][(x + sign*offset[0]) % image[1].shape[0]][(y + offset[1]) % image[1].shape[0]] = color[1]
        image[2][(x + sign*offset[0]) % image[2].shape[0]][(y + offset[1]) % image[2].shape[0]] = color[2]
    return image

