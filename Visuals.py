import os
import sys
import json
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from tqdm import tqdm

def create_visualization_grid(data: np.array, 
                              filename: str = 'animation', 
                              duration: int = 100,  
                              gif: bool = False, 
                              video: bool = False) -> None:
    
    if not gif and not video:
        raise ValueError('At least one of gif or video must be True')
    
    sizes = np.shape(data[0])
    fig = plt.figure()
    fig.set_size_inches(1. * sizes[0] / sizes[1], 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    

    def update(frame):
        ax.clear()
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True        
        ax.set_axis_off()

        ax.axis('off')
        fig.add_axes(ax)
        ax.imshow(data[frame], cmap='Greys', norm=plt.Normalize(0, 1))
        
    # Creating the animation object
    ani = plt.matplotlib.animation.FuncAnimation(
        fig, update, frames=len(data), interval=duration, repeat=False
    )
    
    # Save as GIF if required
    if gif:
        gif_path = f"{filename}.gif"
        ani.save(gif_path, writer=PillowWriter(fps=1000//duration))
        print(f"GIF saved as {gif_path}")
    
    # Save as video if required
    if video:
        video_path = f"{filename}.mp4"
        ani.save(video_path, writer=FFMpegWriter(fps=1000//duration))
        print(f"Video saved as {video_path}")
    
    plt.close(fig)


def visualize_target_result(target, data, filename):            
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(target, cmap='gray')
    ax[0].set_title('Target')
    ax[1].imshow(data[-1], cmap='gray')
    ax[1].set_title('Result')
    ax[0].axis('off')
    ax[1].axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_evolution_results(result_path:str, filename:str):
    with open(result_path, 'r') as f:
        results = json.load(f)
    rewards = results['REWARDS']
    mean_rewards = np.mean(rewards, axis=1)
    std_rewards = np.std(rewards, axis=1)
    plt.figure()
    plt.plot(mean_rewards)
    #fill between
    plt.fill_between(range(len(mean_rewards)), mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    #plot max rewards
    best_rewards = np.max(rewards, axis=1)
    plt.plot(best_rewards, 'r')
    plt.title('Best rewards')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_frame(data, filename, cmap='Blues'):
    plt.figure()
    plt.imshow(data, cmap=cmap)
    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    pass
    #data = [np.random.randint(0, 2, (3, 3, 3)) for _ in range(50)]  # Example data
    #create_visualization_pyvista(data, 'test_video', 100, 'Test Video', gif=False, video=True, rotate=True)
    #data = [np.random.randint(0, 2, (3, 3)) for _ in range(10)]  # Example data
    #create_visualization_grid(data, 'test_video', 100, gif=False, video=True)