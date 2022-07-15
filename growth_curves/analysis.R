library(ggplot2)
library(readr)
library(dplyr)
library(hms)
library(tidyr)

# Import data
dat_raw <- readxl::read_xlsx("20220429_auxin_Plate.xlsx", col_types = "numeric")


dat <-
  dat_raw %>%
  mutate(Time = hms(days = Time)) %>%
  select("Time", matches("[B-G]([2-9]$|1[0-1]$)"))  # Take out columns A, H, rows 1, 12


dat %>% 
  pivot_longer(-Time, names_to = "well", values_to = "OD") %>%
  separate(well, sep = 1, into = c("row", "column")) %>%
  mutate(column = as.factor(as.numeric(column))) %>%
  ggplot(aes(Time, log2(OD))) +
  facet_grid(row ~ column) +
  geom_point(size=0.1)




compute_logODbase <- function(ODs) {
  baseOD <- mean(ODs[1:10])
  log2(ODs - baseOD)
}

X <- cbind(1, dat$Time/3600)  # design matrix
Y <- sapply(dat[, -1], compute_logODbase)

# Find point closest to logODbase = -2
midpoints <- apply(abs(Y + 2), 2, which.min)
midpoints[midpoints < 26] <- 26
midpoints[midpoints > (792 - 25)] <- 792 - 25

idx <- sapply(midpoints, \(x) (x-25):(x+25))
Y_window <- sapply(1:60, \(i) Y[idx[, i], i])
colnames(Y_window) <- colnames(Y)
X_window <- X[1:51, ]

coeffs <- 
  solve(t(X_window) %*% X_window) %*% t(X_window) %*% Y_window %>%
  t %>%
  as_tibble(rownames = "Well") %>%
  transmute(
    Well = Well,
    intercept = V1 - X[midpoints-25, 2] * V2,
    slope = V2
  ) %>%
  separate(Well, sep = 1, into = c("row", "col")) %>%
  mutate(
    row = as.factor(row),
    col = as.factor(as.numeric(col)),
    t_ODthresh = X[midpoints, 2]
  ) %>%
  filter(col != 11) %>%
  arrange(col) %>%
  mutate(
    t_doubl = 1/slope,
    Strain = factor(rep(c("yWT03a", "yCS03a", "yMG01a"), each = 6, times = 3)),
    Replicate = factor(rep(1:3, each = 18)),
    Auxin = factor(rep(c("- auxin", "+ auxin"), each = 3, times =  9))
  ) %>%
  mutate(Strain = factor(Strain, levels = c("yWT03a", "yMG01a", "yCS03a")))
  

data_sub <- dat[seq(1, nrow(dat), by = 10),]

p_curves <- data_sub %>%
  #mutate(across(-Time, compute_logODbase)) %>%
  mutate(Time = as.numeric(Time)/3600) %>%
  #pivot_longer(-Time, names_to = "Well", values_to = "logODbase") %>%
  pivot_longer(-Time, names_to = "Well", values_to = "OD600") %>%
  separate(Well, sep = 1, into = c("row", "col"), remove = FALSE) %>%
  mutate(
    col = as.factor(as.numeric(col)),
    Auxin = ifelse(row %in% c("B", "C", "D"), "- auxin", "+ auxin"),
    Strain = rep(c(rep(c("yWT03a", "yCS03a", "yMG01a"), times = 3), "Ctrl"), 6*nrow(data_sub))
  ) %>%
  mutate(Strain = factor(Strain, levels = c("yWT03a", "yMG01a", "yCS03a"))) %>%
  filter(col != 11) %>%
  #ggplot(aes(Time, logODbase, color = Auxin)) +
  ggplot(aes(Time, OD600, color = Auxin, group = Well)) +
  facet_grid(~ Strain) +
  #geom_point(size = 0.01) +
  geom_line() +
  #geom_abline(data = coeffs, aes(intercept = intercept, slope = slope), linetype = "dashed", lwd = 1, alpha = 0.3) +
  #geom_vline(data = coeffs, aes(xintercept = t_ODthresh)) +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(panel.grid = element_blank(), legend.position = "none", panel.background = element_blank()) +
  xlab("t (hours)") +
  #ylab("log2(OD_base)")
  ylab("OD600")
p_curves

ggsave("growth_curves_OD_pres.pdf", p_curves, device = "pdf", width = 6, height = 2.5)



p_doubling <- coeffs %>%
  mutate(Strain = factor(Strain, levels = c("yMG01a", "yWT03a", "yCS03a"))) %>%
  mutate(t_doubl = t_doubl*6) %>%  # *6 instead of 60 to make axis numbers line up with p_lag (add extra 0s in post-processing)
  ggplot(aes(Strain, t_doubl, fill = Auxin)) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +
  geom_point(
    size = 0.8,
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.5),
    color = "black"
  ) +
  ylim(c(0, 45)) +
  ylab("Doubling time (min)") +
  scale_fill_brewer(palette = "Set1") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    legend.position = c(1, 1),
    legend.justification = c(1, 1),
    legend.background = element_blank()
  )
p_doubling
ggsave("doubling_time_pres.pdf", p_doubling, device = "pdf", width = 3, height = 3)

p_lag <- coeffs %>%
  mutate(Strain = factor(Strain, levels = c("yMG01a", "yWT03a", "yCS03a"))) %>%
  ggplot(aes(Strain, t_ODthresh, fill = Auxin)) +
  geom_boxplot(outlier.shape = NA, width = 0.5) +
  geom_point(
    size = 0.8,
    position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.5),
    color = "black"
  ) +
  ylim(c(0, 75)) +
  scale_fill_brewer(palette = "Set1") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    legend.position = c(1, 1),
    legend.justification = c(1, 1),
    legend.background = element_blank()
  ) +
  ylab("Time until OD600 = 0.5 (hours)")
p_lag
ggsave("lagtime.pdf", p_lag, device = "pdf", width = 3, height = 3)


p_curves_OD <- 
  dat %>%
  mutate(Time = as.numeric(Time)/3600) %>%
  pivot_longer(-Time, names_to = "Well", values_to = "OD") %>%
  separate(Well, sep = 1, into = c("row", "col")) %>%
  mutate(
    col = as.factor(as.numeric(col)),
    Auxin = ifelse(row %in% c("B", "C", "D"), "- auxin", "+ auxin"),
    Strain = rep(c(rep(c("yCS02a", "yCS03a", "yMG01a"), times = 3), "Ctrl"), 4752)
  ) %>%
  filter(col != 11) %>%
  ggplot(aes(Time, OD, color = Auxin)) +
  facet_grid(Auxin ~ Strain) +
  geom_point(size = 0.1) +
  #geom_vline(data = coeffs, aes(xintercept = t_ODthresh)) +
  scale_color_brewer(palette = "Set1") +
  theme_bw() +
  theme(panel.grid = element_blank(), legend.position = "none", panel.background = element_blank()) +
  xlab("t (hours)") +
  ylab("OD600")

ggsave("growth_curves.pdf", p_curves_OD, device = "pdf", width = 6, height = 4)


coeffs %>%
  mutate(t_doubl = t_doubl*60) %>%
  group_by(Strain, Auxin) %>%
  summarize(
    Mean2 = round(mean(t_ODthresh), 1),
    SE2 = round(sd(t_ODthresh)/sqrt(n()), 1),
    Mean = round(mean(t_doubl), 0),
    SE = round(sd(t_doubl)/sqrt(n()), 0)
  ) %>%
  write_csv('summary_growth.csv')

coeffs %>%
  group_by(Strain, Auxin) %>%
  summarize(
    Mean = mean(t_ODthresh),
    SE = sd(t_ODthresh)/sqrt(n())
  )
