{-# LANGUAGE CPP #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}

{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FlexibleInstances #-}

{-# LANGUAGE PolyKinds #-}

{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE Rank2Types #-}

{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}

-- To automatically derive Binary
{-# LANGUAGE DeriveAnyClass #-}


module Net.ArcGraph
  ( 
  -- * Network
--     Config(..)
    Param(..)
  -- , printParam
  -- , size
  , new
  , Graph(..)
  , Node(..)
  , Arc
  , run
  , eval
  , runQ
  , evalQ
  -- , runTest

  -- * Opaque
  , Opaque(..)
  , Typ(..)
  , newO

  -- * Storage
  , saveParam
  , loadParam

  -- * Data set
  , DataSet
  , Elem(..)
  -- , mkGraph
  -- , mkSent

  -- * Error
  , netError
  , netErrorQ

  -- * Training
  -- , trainProg

--   -- * Test
--   , Test
--   , Opaque (..)
--   , test1
--   , runOpaque
  ) where


import           GHC.Generics (Generic)
import           GHC.TypeNats (KnownNat, natVal)
-- import           GHC.Natural (naturalToInt)
import qualified GHC.TypeNats as Nats
import           GHC.TypeLits (Symbol, KnownSymbol)

import           System.Random (randomRIO)

import           Control.Monad (forM_, forM)

import           Lens.Micro.TH (makeLenses)
-- import           Lens.Micro ((^.))

import           Data.Proxy (Proxy(..))
-- import           Data.Ord (comparing)
-- import qualified Data.List as L
import           Data.Maybe (catMaybes, mapMaybe)
import qualified Data.Text as T
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.Graph as G
import qualified Data.Array as A
import           Data.Binary (Binary)
import qualified Data.Binary as Bin
import qualified Data.ByteString.Lazy as BL
import           Codec.Compression.Zlib (compress, decompress)

-- import qualified Text.Read as R
-- import qualified Text.ParserCombinators.ReadP as R

-- import qualified Data.Map.Lens as ML
import           Control.Lens.At (ixAt)
import           Control.Lens (Lens)
import           Control.DeepSeq (NFData)

import           Dhall (Interpret)
import qualified Data.Aeson as JSON

import qualified Prelude.Backprop as PB
import qualified Numeric.Backprop as BP
import           Numeric.Backprop (Backprop, (^^.), (^^?))
import qualified Numeric.LinearAlgebra.Static.Backprop as LBP
import           Numeric.LinearAlgebra.Static.Backprop
  (R, L, BVar, Reifies, W, (#), (#>), dot)
import qualified Numeric.LinearAlgebra.Static as LA

import           Net.Basic
-- import qualified Net.List as NL
import qualified Net.FeedForward as FFN
import           Net.FeedForward (FFN(..))
-- import qualified GradientDescent as GD
-- import qualified GradientDescent.Momentum as Mom
-- import qualified GradientDescent.Nestorov as Mom
-- import qualified GradientDescent.AdaDelta as Ada
import           Numeric.SGD.ParamSet (ParamSet)
import qualified Numeric.SGD.ParamSet as SGD

import           Net.ArcGraph.Graph
import qualified Net.ArcGraph.BiComp as B
import qualified Net.ArcGraph.QuadComp as Q

-- import qualified Embedding.Dict as D

import           Debug.Trace (trace)


-- | Model type
data Typ
  = Arc0T
  | Arc1T
  | Arc2T
  | Arc3T
  | Arc4T
  | Arc5T
  | Arc6T
  | Quad0T
  | Quad1T
  deriving (Generic, Binary, Read)


-- instance Read Typ where
--   readPrec =
--     Arc0T <$ str "arc0"
--     Arc1T <$ str "arc1"
--     Arc2T <$ str "arc2"
--     where
--       str = R.lift . R.string
--   readListPrec = R.readListPrecDefault


-- | Execute the given action within the context of a particular model type.
exec
  :: forall d a b m.
    ( KnownNat d
    , Binary a, NFData a, Show a, Ord a
    , Binary b, NFData b, Show b, Ord b
    , Functor m
    )
  => Typ
  -> (forall p. (Binary p, New a b p) => m p)
  -> m (Opaque d a b)
exec typ action =
  case typ of
    Arc0T -> Opaque typ <$> (action :: m (Arc0 d a b))
    Arc1T -> Opaque typ <$> (action :: m (Arc1 d a b))
    Arc2T -> Opaque typ <$> (action :: m (Arc2 d a b))
    Arc3T -> Opaque typ <$> (action :: m (Arc3 d a b))
    Arc4T -> Opaque typ <$> (action :: m (Arc4 d a b))
    Arc5T -> Opaque typ <$> (action :: m (Arc5 d a b))
    Arc6T -> Opaque typ <$> (action :: m (Arc6 d a b))
    Quad0T -> Opaque typ <$> (action :: m (Quad0 d a b))
    Quad1T -> Opaque typ <$> (action :: m (Quad1 d a b))


-- | Opaque parameter set (with the actual quad component abstracted away).
-- The type (`Typ`) is kept for the sake of (de)serialization.
data Opaque :: Nats.Nat -> * -> * -> * where
  Opaque 
    :: (Binary p, NFData p, SGD.ParamSet p, Q.QuadComp d a b p)
    => Typ -> p -> Opaque d a b


instance ( KnownNat d
         , Binary a, Binary b, NFData a, NFData b
         , Ord a, Ord b, Show a, Show b
         )
  => Binary (Opaque d a b) where
  put (Opaque typ p) = Bin.put typ >> Bin.put p
  get = do
    typ <- Bin.get
    exec typ Bin.get


newO
  :: forall d a b.
    ( KnownNat d
    , Binary a, NFData a, Show a, Ord a
    , Binary b, NFData b, Show b, Ord b
    )
  => Typ
  -> S.Set a
  -> S.Set b
  -> IO (Opaque d a b)
newO typ xs ys =
  exec typ (new xs ys)


----------------------------------------------
-- Parameters
----------------------------------------------


-- | Possible model configurations
data Param d a b 
  = PArc0 (Arc0 d a b)
  | PArc1 (Arc1 d a b)
  | PArc2 (Arc2 d a b)
  | PArc3 (Arc3 d a b)
  | PArc4 (Arc4 d a b)
  | PQuad0 (Quad0 d a b)
  | PQuad1 (Quad1 d a b)
  deriving (Generic, Binary)


-- | Arc-factored model (0)
type Arc0 d a b
   = Q.BiAff d 100
  :& Q.Bias


-- | Arc-factored model (1)
type Arc1 d a b
   = Q.BiAff d 100
  :& Q.BiQuad (BiParam a b)


-- | Arc-factored model (2)
type Arc2 d a b
   = Q.BiAffExt d 50 a b 100
  :& Q.BiQuad (BiParam a b)


-- | Arc-factored model (3)
type Arc3 d a b
  = Q.BiQuad (PotArc d 100 a b)


-- | Arc-factored model (4); similar to (3), but with (h = d).
-- The best for LVC.full in French found so far!
type Arc4 d a b
  = Q.BiQuad (PotArc d d a b)


-- | Arc-factored model (5): (2) - `BiParam`
type Arc5 d a b
   = Q.BiAffExt d 50 a b 100


-- | Arc-factored model (6): (2) + `Q.UnordBiAff`
type Arc6 d a b
   = Q.BiAffExt d 50 a b 100
  :& Q.UnordBiAff d 100
  :& Q.BiQuad (BiParam a b)


-- | Basic bi-affine compoments (easy to compute, based on POS and DEP labels
-- exclusively)
type BiParam a b 
   = B.Bias
  :& B.HeadPosAff a
  :& B.DepPosAff a
  :& B.PapyPosAff a
  :& B.EnkelPosAff a
  :& B.ArcBias b
  :& B.PapyArcAff b
  :& B.EnkelArcAff b


-- | The best arc-factored model you found so far (with h = dim).
type PotArc dim h a b
   = B.Biaff dim h
  :& B.UnordBiaff dim h
  :& B.HeadAff dim h
  :& B.DepAff dim h
  :& BiParam a b


-- | Quad-factored model (0) with hidden layers of size 100
type Quad0 d a b = 
  QuadH d 100 a b


-- | Quad-factored model (1); (0) + unordered bi-affine component.  That's the
-- last model you tried on cl-srv2.  Turned out better than `Quad0`, the
-- `Q.UnordBiAff` seems to make a difference.
--
-- Now you could try to do some ablation/enriching study:
--
--   * What if you remove `Q.TriAff` and `Q.SibAff`?
--   * What if you also remove `Q.UnAff`?
--
-- You can also try `Arc3`, a simplified (with dim = 100) version of the best
-- model obtained so far (`Arc4`).  Just to see how much you potentialy (well,
-- it can depend on training anyway...) lose by using smaller dimension.
--
-- The next thing to do would be to check if you can gain somethin by using
-- POS/DEP embeddings.  For instance by enriching `Arc3` or `Arc4` with
-- `Q.BiAffExt` (see `Arc2`).  In fact, it makes sense to test `Arc1`, `Arc2`,
-- and `Arc3` first and see what they give.
--
type Quad1 d a b
   = QuadH d 100 a b
  :& Q.UnordBiAff d 100


-- | Quad-factored model with underspecified size of the hidden layers
type QuadH d h a b
   = Q.TriAff d h
  :& Q.SibAff d h
  :& Q.BiAff d h
  :& Q.UnAff d h
  :& Q.Bias


----------------------------------------------
-- Evaluation
----------------------------------------------


run
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , B.BiComp dim a b comp
     )
  => BVar s comp
    -- ^ Graph network / params
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
run net graph = M.fromList $ do
  arc <- graphArcs graph
  let x = B.runBiComp graph arc net
  return (arc, logistic x)


-- | Evaluate the network over a graph labeled with input word embeddings.
-- User-friendly (and without back-propagation) version of `run`.
eval
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
eval net graph =
  BP.evalBP0
    ( BP.collectVar
    $ run
        (BP.constVar net)
        graph
    )


runQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , Q.QuadComp dim a b comp
     )
  => BVar s comp
    -- ^ Quad component
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states
  -> M.Map Arc (BVar s Double)
    -- ^ Output map with output potential values
runQ net graph = normalize . M.unionsWith (+) $ do
  arc <- graphArcs graph
  return $ Q.runQuadComp graph arc net
  where
    normalize = fmap logistic


evalQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b
     , Q.QuadComp dim a b comp
     )
  => comp
    -- ^ Graph network / params
  -- -> Graph (R d) b
  -> Graph (Node dim a) b
    -- ^ Input graph labeled with initial hidden states (word embeddings)
  -> M.Map Arc Double
evalQ net graph =
  BP.evalBP0
    ( BP.collectVar
    $ runQ
        (BP.constVar net)
        graph
    )


----------------------------------------------
-- Serialization
----------------------------------------------


-- | Save the parameters in the given file.
saveParam :: (Binary a) => FilePath -> a -> IO ()
saveParam path =
  BL.writeFile path . compress . Bin.encode


-- | Load the parameters from the given file.
loadParam :: (Binary a) => FilePath -> IO a
loadParam path =
  Bin.decode . decompress <$> BL.readFile path


----------------------------------------------
-- DataSet
----------------------------------------------


-- | Dataset element
data Elem dim a b = Elem
  { graph :: Graph (Node dim a) b
    -- ^ Input graph
  , labMap :: M.Map Arc Double
    -- ^ Target probabilities
  } deriving (Show, Generic, Binary)


-- | DataSet: a list of dataset elements
type DataSet d a b = [Elem d a b]


----------------------------------------------
-- Error
----------------------------------------------


-- -- | Squared error between the target and the actual output.
-- errorOne
--   :: (Ord a, KnownNat n, Reifies s W)
--   => M.Map a (BVar s (R n))
--     -- ^ Target values
--   -> M.Map a (BVar s (R n))
--     -- ^ Output values
--   -> BVar s Double
-- errorOne target output = PB.sum . BP.collectVar $ do
--   (key, tval) <- M.toList target
--   let oval = output M.! key
--       err = tval - oval
--   return $ err `dot` err


-- | Cross entropy between the true and the artificial distributions
crossEntropy
  :: (Reifies s W)
  => BVar s Double
    -- ^ Target ,,true'' MWE probability
  -> BVar s Double
    -- ^ Output ,,artificial'' MWE probability
  -> BVar s Double
crossEntropy p q = negate
  ( p0 * log q0
  + p1 * log q1
  )
  where
    p1 = p
    p0 = 1 - p1
    q1 = q
    q0 = 1 - q1


-- -- | Cross entropy between the true and the artificial distributions
-- crossEntropy
--   :: (KnownNat n, Reifies s W)
--   => BVar s (R n)
--     -- ^ Target ,,true'' distribution
--   -> BVar s (R n)
--     -- ^ Output ,,artificial'' distribution
--   -> BVar s Double
-- crossEntropy p q =
--   negate (p `dot` LBP.vmap' log q)


-- | Cross entropy between the target and the output values
errorOne
  :: (Ord a, Reifies s W)
--   => M.Map a (BVar s (R n))
  => M.Map a (BVar s Double)
    -- ^ Target values
--   -> M.Map a (BVar s (R n))
  -> M.Map a (BVar s Double)
    -- ^ Output values
  -> BVar s Double
errorOne target output = PB.sum . BP.collectVar $ do
  (key, tval) <- M.toList target
  let oval = output M.! key
  return $ crossEntropy tval oval


-- | Error on a dataset.
errorMany
  :: (Ord a, Reifies s W)
--   => [M.Map a (BVar s (R n))] -- ^ Targets
--   -> [M.Map a (BVar s (R n))] -- ^ Outputs
  => [M.Map a (BVar s Double)] -- ^ Targets
  -> [M.Map a (BVar s Double)] -- ^ Outputs
  -> BVar s Double
errorMany targets outputs =
  go targets outputs
  where
    go ts os =
      case (ts, os) of
        (t:tr, o:or) -> errorOne t o + go tr or
        ([], []) -> 0
        _ -> error "errorMany: lists of different size"


-- | Network error on a given dataset.
netError
  :: ( Reifies s W, KnownNat dim
     , Ord a, Show a, Ord b, Show b
     , B.BiComp dim a b comp
     )
  => DataSet dim a b
  -- -> BVar s (Param dim a b)
  -> BVar s comp
  -> BVar s Double
netError dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (run net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)


-- | Network error on a given dataset.
netErrorQ
  :: ( KnownNat dim, Ord a, Show a, Ord b, Show b, Reifies s W
     , Q.QuadComp dim a b comp
     )
  => DataSet dim a b
  -> BVar s comp
  -> BVar s Double
netErrorQ dataSet net =
  let
    inputs = map graph dataSet
    -- outputs = map (run net . nmap BP.auto) inputs
    outputs = map (runQ net) inputs
    targets = map (fmap BP.auto . labMap) dataSet
  in  
    errorMany targets outputs -- + (size net * 0.01)
