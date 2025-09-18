def plot(mlip):

    fig, ax = plt.subplots(figsize=(6.5,6.5))
    ax.plot([0.1, 2*10**3], [0.1, 2*10**3], color='gray', linestyle='--' ,linewidth=1, zorder=0)

    for i in range(len(df_wz)):
        ax.scatter(df_wz.iloc[i]['exp'], df_wz.iloc[i][mlip], marker='^', s=300, edgecolors='k', linewidths=1.1, alpha = 0.8)
    for i in range(len(df_zb)):
        ax.scatter(df_zb.iloc[i]['exp'], df_zb.iloc[i][mlip], marker='s', s=230, edgecolors='k', linewidths=1.1, alpha = 0.8)
        
    r2 = r2_score(df['exp'], df[mlip])
    mae = mean_absolute_error(df['exp'], df[mlip])
    ax.text(0.04, 0.96, mlip, transform=ax.transAxes, fontsize=20,
            verticalalignment='top', horizontalalignment='left')
    ax.text(0.04, 0.90, f'R$^2$={r2:.2f}\nMAE={mae:.2f}', transform=ax.transAxes, fontsize=18,
            verticalalignment='top', horizontalalignment='left',)


    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r"κ experiment (W/m$\cdot$K)")
    ax.set_ylabel(r"κ MLIP (W/m$\cdot$K)")
    ax.set_xlim(0.4, 2*10**3)
    ax.set_ylim(0.4, 2*10**3)
    # ax.legend(frameon=True, fontsize=12,markerscale=1.2, handletextpad=0.3, borderpad=0.5, labelspacing=0.8, bbox_to_anchor=(0.02, 0.98), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'rta_exp_{mlip}.png', dpi=300)
    plt.show()