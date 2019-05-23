{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}

{-# OPTIONS_GHC -fno-warn-orphans #-}


-- | The top-level module of the VMWE identification tool.  Provides plumbing
-- between various more specialized modules:
--
--   * `N.Graph`     -- Labelled graphs
--   * `Net.Graph`   -- Graph neural network
--   * `Format.Cupt` -- .cupt format support
--   * `Numeric.SGD` -- stochastic gradient descent (external library)
--   * ...


module MWE
  ( 
  -- * Types
    Sent(..)

  -- * New
  , N.new

  -- * Training
  , Config(..)
  , Method(..)
  , train
  , depRelsIn
  , posTagsIn

  -- * Tagging
  , TagConfig(..)
  , tagManyT
  ) where


import           Prelude hiding (elem)

import           GHC.Generics (Generic)

import           Control.Monad (forM_)
import           Control.Parallel.Strategies (parMap, rseq)

import           Lens.Micro ((^.))

import qualified Numeric.Backprop as BP

import           Dhall (Interpret(..), genericAuto)

import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy as L

import qualified Numeric.SGD as SGD
import qualified Numeric.SGD.Type as SGD
import qualified Numeric.SGD.ParamSet as SGD
import qualified Numeric.SGD.DataSet as SGD
import qualified Numeric.SGD.Momentum as Mom
import qualified Numeric.SGD.AdaDelta as Ada
import qualified Numeric.SGD.Adam as Adam

import qualified Format.Cupt as Cupt
import qualified Net.Graph as N
import qualified DhallUtils as DU
import qualified MWE.Sent as Sent
import           MWE.Sent (Sent(..))
import           MWE.Anno (annotate)

-- import Debug.Trace (trace)


----------------------------------------------
-- Training
----------------------------------------------


-- Some orphan `Interpret` instances
instance Interpret Mom.Config
instance Interpret Ada.Config
instance Interpret Adam.Config
instance Interpret SGD.Config


data Method
  = Momentum Mom.Config
  | AdaDelta Ada.Config
  | Adam Adam.Config
  deriving (Generic)

instance Interpret Method where
  autoWith _ = DU.rmUnion_1 genericAuto


data Config = Config
  { sgd :: SGD.Config
  , method :: Method
  , probCfg :: N.Config
  } deriving (Generic)

instance Interpret Config


-- | Select the sgd method
toSGD
  :: (SGD.ParamSet p)
  => Config
    -- ^ Configuration
  -> Int
    -- ^ Dataset size
  -> (e -> p -> p)
    -- ^ Gradient
  -> SGD.SGD IO e p
toSGD Config{..} dataSize grad =
  case method of
    Momentum cfg -> Mom.momentum (Mom.scaleTau iterNumEpoch cfg) grad
    AdaDelta cfg -> Ada.adaDelta cfg grad
    Adam cfg -> Adam.adam (Adam.scaleTau iterNumEpoch cfg) grad
  where
    iterNumEpoch = SGD.iterNumPerEpoch sgd dataSize


-- | Depdency relation
type DepRel = T.Text


-- | POS tag
type POS = T.Text


-- | Extract dependeny relations present in the given dataset.
depRelsIn :: [Cupt.GenSent mwe] -> S.Set DepRel
depRelsIn =
  S.fromList . (Sent.dummyRootDepRel:) . concatMap extract 
  where
    extract = map Cupt.deprel


-- | Extract dependeny relations present in the given dataset.
posTagsIn :: [Cupt.GenSent mwe] -> S.Set POS
posTagsIn =
  S.fromList . (Sent.dummyRootPOS:) . concatMap extract 
  where
    extract = map Cupt.upos


-- | Train the MWE identification network.
train
  :: Config
    -- ^ General training confiration
  -> Cupt.MweTyp
    -- ^ Selected MWE type (category)
  -> [Sent 300]
    -- ^ Training dataset
  -> N.Transparent
    -- ^ Initial network
  -> (N.Transparent -> IO ())
    -- ^ Action to execute at the end of each epoch
  -> IO N.Transparent
train cfg mweTyp cupt tra0 action = do
  let cupt' = map (Sent.mkElem (== mweTyp)) cupt
  SGD.withDisk cupt' $ \dataSet -> do
    putStrLn $ "# Training dataset size: " ++ show (SGD.size dataSet)
    SGD.runIO (sgd cfg)
      (toSGD cfg (SGD.size dataSet) (SGD.batchGradPar gradient))
      (reportAndExec dataSet) dataSet tra0
  where
    gradient x = BP.gradBP (N.netError (probCfg cfg) x)
    quality x = BP.evalBP (N.netError (probCfg cfg) x)
    reportAndExec dataSet tra = do
      x <- SGD.reportObjective quality dataSet tra
      action tra
      return x


----------------------------------------------
-- Tagging
----------------------------------------------


-- | Tagging configuration
data TagConfig = TagConfig
  { mweTyp :: Cupt.MweTyp
    -- ^ MWE category to annotate
  , mweConstrained :: Bool
    -- ^ Constrained global decoding
  } deriving (Show, Eq, Ord)


-- | Tag a single sentence with the given network.
tag
  :: TagConfig
  -> N.Transparent   -- ^ Network parameters
  -> Sent 300        -- ^ Cupt sentence
  -> Cupt.Sent
tag tagCfg net sent =
  sent'
  where
    elem = N.evalInp (Sent.mkElem (const False) sent) net
    tagF
      | mweConstrained tagCfg =
          N.treeTagConstrained (N.graph elem)
      | otherwise =
          N.treeTagGlobal (N.graph elem)
    labeling = tagF
      (N.evalBia (net ^. N.biaMod) (N.graph elem))
      (N.evalUni (net ^. N.uniMod) (N.graph elem))
    sent' = annotate (mweTyp tagCfg) (cuptSent sent) labeling


-- | Tag a single sentence with the given network.
tagToText
  :: TagConfig
  -> N.Transparent   -- ^ Network parameters
  -> Sent 300        -- ^ Cupt sentence
  -> T.Text
tagToText tagCfg net sent =
  L.toStrict $ Cupt.renderPar [Cupt.abstract sent']
  where
    sent' = tag tagCfg net sent


-- | Tag and annotate sentences in parallel.
tagManyTpar
  :: TagConfig
  -> N.Transparent
  -> [Sent 300]
  -> [T.Text]
tagManyTpar cfg net = parMap rseq (tagToText cfg net)


-- | Tag sentences with the opaque network.
tagManyT
  :: TagConfig
  -> N.Transparent
  -> [Sent 300]
  -> IO ()
tagManyT cfg net cupt0 = do
  forM_ (zip cupt0 cupt) $ \(sent0, sent) -> do
    T.putStr "# "
    T.putStrLn . T.unwords . map Cupt.orth $ cuptSent sent0
    T.putStrLn sent
  where
    cupt = tagManyTpar cfg net cupt0
